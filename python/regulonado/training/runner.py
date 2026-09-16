from __future__ import annotations

import dataclasses
import json
import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence
from os import environ
from pathlib import Path
from time import perf_counter
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from torch.optim import AdamW
from torch.utils.data import Subset
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
    EvalExampleDiagnostics,
    LRLogCallback,
    PerTrackMetricsReport,
    WandbConfigCallback,
)
from regulonado.training.config import TrainerConfig
from regulonado.training.data import WindowParquetDataset
from regulonado.training.losses import (
    contrast_group_weights,
    kendall_track_weighted_loss,
    log1p_huber_loss,
    poisson_multinomial_binwise_loss,
    poisson_multinomial_loss,
    poisson_nll_loss,
    scaled_poisson_multinomial_loss,
    topk_additive_loss,
    topk_reweight_loss,
    track_contrast_correlation_loss,
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


def _world_size() -> int:
    """Number of training processes under torchrun/DDP (1 when launched single-process)."""
    return int(environ.get("WORLD_SIZE") or 1)


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
    # track_loss_log_var is loss state registered only when loss.learn_track_weights is
    # set, so phases may legitimately differ: a fresh one starts at zero, a stale one is
    # dropped.
    missing = [k for k in missing if k != "track_loss_log_var"]
    unexpected = [k for k in unexpected if k != "track_loss_log_var"]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys when warm-starting: {unexpected[:10]}")
    if missing:
        raise RuntimeError(f"Missing checkpoint keys when warm-starting: {missing[:10]}")


def assert_track_names_match(
    model_track_names: Sequence[str], dataset_track_names: Sequence[str]
) -> None:
    """Reject equal-sized but differently ordered output channels."""
    model_names = list(model_track_names)
    dataset_names = list(dataset_track_names)
    if model_names and model_names != dataset_names:
        raise ValueError(
            "Model and dataset track order differs; refusing to silently permute head channels: "
            f"model={model_names!r}, dataset={dataset_names!r}"
        )


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


def _usable_group_label(value: Any) -> str | None:
    """Return a canonical freeform group label, or None for an unusable value."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    label = str(value).strip()
    return label or None


def resolved_condition_ids(
    records: Sequence[Mapping[str, Any]], condition_source: str = "condition_id"
) -> list[int]:
    """Resolve the categorical condition IDs passed to metadata-aware heads.

    ``group`` values are intentionally opaque labels. Sorting their distinct
    normalized values makes the encoding stable across track-table row order.
    """
    if condition_source == "condition_id":
        return _track_array(records, "condition_id", dtype=np.int64, fill_value=-1).tolist()
    if condition_source != "group":
        raise ValueError(
            f"model.condition_source must be 'condition_id' or 'group', got {condition_source!r}"
        )

    labels = [_usable_group_label(record.get("group")) for record in records]
    vocabulary = sorted({label for label in labels if label is not None})
    if not vocabulary:
        raise ValueError(
            "model.condition_source='group' requires at least one non-empty "
            "'group' value in tracks.parquet"
        )
    encoded = {label: index for index, label in enumerate(vocabulary)}
    return [encoded[label] if label is not None else -1 for label in labels]


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


def constant_track_metadata_values(
    records: Sequence[Mapping[str, Any]], *, condition_source: str = "condition_id"
) -> dict[str, list]:
    values: dict[str, list] = {}
    for out_key, keys in _TRACK_METADATA_FIELD_MAP.items():
        if out_key == "track_condition_ids":
            values[out_key] = resolved_condition_ids(records, condition_source)
            continue
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


def constant_track_metadata(
    records: Sequence[Mapping[str, Any]], *, condition_source: str = "condition_id"
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for out_key, values in constant_track_metadata_values(
        records, condition_source=condition_source
    ).items():
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
        for index, example in enumerate(batch):
            missing = sorted({"input_ids", "labels"} - example.keys())
            if missing:
                raise ValueError(
                    f"Training example {index} is missing transformed field(s): "
                    f"{', '.join(missing)}; available fields: {sorted(example)}"
                )
        collated = {
            "input_ids": torch.stack([torch.as_tensor(example["input_ids"]) for example in batch]),
            "labels": torch.stack(
                [torch.as_tensor(example["labels"]) for example in batch]
            ).float(),
        }
        for key, value in constant_tensors.items():
            collated[key] = value
        return collated

    return collate


def _contrast_weights_from_records(records: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    """Per-group replicate-averaging contrast weights keyed by ``assay_class`` and ``group``."""
    return contrast_group_weights(
        [record.get("assay_class") for record in records],
        [record.get("group") for record in records],
    )


def _build_loss_fn(
    loss_cfg: Mapping[str, Any],
    *,
    scale_factors: np.ndarray,
    clip_hard: np.ndarray,
    labels_already_scaled: bool,
    contrast_weights: torch.Tensor | None = None,
    track_log_var: torch.nn.Parameter | None = None,
    contrast_region_bins: int = 16,
    contrast_pseudocount: float = 0.1,
    contrast_active_fraction: float = 0.1,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the configured base loss, plus the cross-track contrast term when weighted."""
    base_loss = _build_base_loss_fn(
        loss_cfg,
        scale_factors=scale_factors,
        clip_hard=clip_hard,
        labels_already_scaled=labels_already_scaled,
        track_log_var=track_log_var,
    )
    contrast_weight = float(loss_cfg.get("contrast_weight") or 0.0)
    if contrast_weight <= 0.0:
        return base_loss
    if contrast_weights is None or contrast_weights.shape[0] == 0:
        raise ValueError(
            "loss.contrast_weight requires tracks labelled with assay_class and group, "
            "with at least two groups sharing one assay_class"
        )
    return lambda pred, target: (
        base_loss(pred, target)
        + contrast_weight
        * track_contrast_correlation_loss(
            pred,
            target,
            contrast_weights,
            region_bins=contrast_region_bins,
            pseudocount=contrast_pseudocount,
            active_fraction=contrast_active_fraction,
        )
    )


def _build_base_loss_fn(
    loss_cfg: Mapping[str, Any],
    *,
    scale_factors: np.ndarray,
    clip_hard: np.ndarray,
    labels_already_scaled: bool,
    track_log_var: torch.nn.Parameter | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    loss_name = str(loss_cfg.get("name", "poisson_multinomial"))
    poisson_weight = float(loss_cfg.get("poisson_weight", 0.2))
    huber_delta = float(loss_cfg.get("delta", 1.0))
    learn_track_weights = bool(loss_cfg.get("learn_track_weights", False))
    if learn_track_weights and loss_name not in {
        "poisson_multinomial",
        "poisson_multinomial_binwise",
    }:
        raise ValueError(
            f"loss.learn_track_weights is only supported for poisson_multinomial and "
            f"poisson_multinomial_binwise, got loss.name={loss_name!r}"
        )
    if learn_track_weights and track_log_var is None:
        raise ValueError(
            "loss.learn_track_weights is set but no track_log_var parameter was supplied"
        )

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
        weight_range = float(loss_cfg.get("weight_range", 0.0))
        weight_exp = float(loss_cfg.get("weight_exp", 1.0))
        if learn_track_weights:
            return lambda pred, target: kendall_track_weighted_loss(
                poisson_multinomial_loss(
                    pred,
                    target,
                    poisson_weight=poisson_weight,
                    weight_range=weight_range,
                    weight_exp=weight_exp,
                    reduction="none",
                ),
                track_log_var,
            )
        return lambda pred, target: poisson_multinomial_loss(
            pred,
            target,
            poisson_weight=poisson_weight,
            weight_range=weight_range,
            weight_exp=weight_exp,
        )
    if loss_name == "poisson_multinomial_binwise":
        if learn_track_weights:
            return lambda pred, target: kendall_track_weighted_loss(
                poisson_multinomial_binwise_loss(
                    pred,
                    target,
                    poisson_weight=poisson_weight,
                    reduction="none",
                ),
                track_log_var,
            )
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
        bin_mode = str(loss_cfg.get("bin_mode", "signal_weighted"))
        bin_signal_power = float(loss_cfg.get("bin_signal_power", 1.0))
        bin_threshold = float(loss_cfg.get("bin_threshold", 0.0))
        topk_bin_weight = float(loss_cfg.get("topk_bin_weight", 0.0))
        topk_bin_count = int(loss_cfg.get("topk_bin_count", 0))
        topk_huber_delta = float(loss_cfg.get("topk_huber_delta", 1.0))
        return lambda pred, target: transfer_calibration_loss(
            pred,
            target,
            profile_weight=profile_weight,
            total_weight=total_weight,
            bin_weight=bin_weight,
            bin_mode=bin_mode,
            bin_signal_power=bin_signal_power,
            bin_threshold=bin_threshold,
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
    dataset_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    data_cfg: Mapping[str, Any],
) -> Mapping[str, Any]:
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
    # WindowParquetDataset applies its `.transform` inside __getitem__, per example.
    dataset_dict["train"].transform = train_transform
    if "validation" in dataset_dict:
        dataset_dict["validation"].transform = eval_transform
    if "test" in dataset_dict:
        dataset_dict["test"].transform = eval_transform
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
    condition_source = str(model_cfg.get("condition_source", "condition_id"))
    if condition_source not in {"condition_id", "group"}:
        raise ValueError(
            f"model.condition_source must be 'condition_id' or 'group', got {condition_source!r}"
        )
    condition_ids = resolved_condition_ids(records, condition_source) if use_track_metadata else []
    target_length = int(metadata.get("n_pred_bins", 0)) or None

    shared_track_index: list[int] = []
    if bool(model_cfg.get("share_condition_base_channels", True)) and use_track_metadata:
        shared_track_index = build_condition_shared_track_index(
            records, condition_source=condition_source
        )

    head_type = str(head_cfg.get("type", "residual_film"))

    # Derive filesystem-safe track names from the resolved BigWig paths.
    track_names: list[str] = []
    seen: dict[str, int] = {}
    for idx, record in enumerate(records):
        path = (
            record.get("track_name")
            or record.get("resolved_path")
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
        output_bias_init=head_cfg.get("resolved_output_bias"),
        zero_output_weights=bool(head_cfg.get("zero_output_weights", False)),
        n_tracks=len(records),
        feature_dim=int(getattr(backbone, "feature_dim", 1920)),
        use_track_metadata=use_track_metadata,
        condition_source=condition_source,
        activation_type=str(model_cfg.get("activation_type", "softplus")),
        num_conditions=(max(condition_ids, default=-1) + 1 if use_track_metadata else 0),
        num_cell_lines=(
            infer_cardinality_any(records, "source_id", "cell_line_id") if use_track_metadata else 0
        ),
        num_assay_types=infer_cardinality(records, "assay_type_id") if use_track_metadata else 0,
        num_targets=infer_cardinality(records, "target_id") if use_track_metadata else 0,
        metadata_hidden=int(model_cfg.get("metadata_hidden", 32)),
        condition_shared_track_index=shared_track_index,
        context_length=int(metadata.get("context_length", data_cfg.get("context_length", 524_288))),
        n_pred_bins=int(metadata.get("n_pred_bins", data_cfg.get("n_pred_bins", 6_144))),
        bin_size=int(metadata.get("bin_size", 32)),
        track_names=track_names,
        track_metadata=(
            constant_track_metadata_values(records, condition_source=condition_source)
            if use_track_metadata
            else {}
        ),
        data_path=str(cfg["data"]["path"]),
    )


def _inverse_output_activation_mean(means: np.ndarray, activation_type: str) -> np.ndarray:
    """Map desired positive output means to final-layer bias values."""
    means = np.maximum(np.asarray(means, dtype=np.float64), 1e-8)
    if activation_type == "exp":
        return np.log(means)
    if activation_type == "softplus_beta2":
        return np.log(np.expm1(2.0 * means)) / 2.0
    if activation_type == "softplus":
        # log(expm1(x)) is unstable for large x, where softplus^-1(x) ~= x.
        return np.where(means > 20.0, means, np.log(np.expm1(means)))
    if activation_type == "identity":
        return means
    raise ValueError(f"Empirical output-bias initialization does not support {activation_type!r}")


def _empirical_track_output_bias(
    train_dataset: Any,
    *,
    n_tracks: int,
    activation_type: str,
    max_samples: int,
    seed: int = 0,
) -> list[float]:
    """Estimate per-track transformed-label means from a seeded random sample of rows.

    Sampling by seeded random index (rather than the first ``max_samples`` rows) avoids
    a genome-order bias: chromosome-major shard order means the first rows of a fresh
    dataset are all chr1.
    """
    if max_samples < 1:
        raise ValueError("head.output_init_samples must be at least 1")
    n_total = len(train_dataset)
    if n_total == 0:
        raise ValueError("Cannot initialize output bias from an empty training dataset")
    rng = np.random.default_rng(seed)
    sample_size = min(max_samples, n_total)
    indices = rng.choice(n_total, size=sample_size, replace=False)

    totals = np.zeros(n_tracks, dtype=np.float64)
    count = 0
    for index in indices:
        labels = np.asarray(train_dataset[int(index)]["labels"], dtype=np.float64)
        if labels.ndim != 2:
            raise ValueError(f"Expected 2D training labels, got shape {labels.shape}")
        if labels.shape[0] != n_tracks and labels.shape[1] == n_tracks:
            labels = labels.T
        if labels.shape[0] != n_tracks:
            raise ValueError(
                f"Training label shape {labels.shape} does not contain {n_tracks} tracks"
            )
        totals += labels.sum(axis=1)
        count += labels.shape[1]
    if count == 0:
        raise ValueError("Cannot initialize output bias from an empty training dataset")
    return _inverse_output_activation_mean(totals / count, activation_type).tolist()


def _resolve_empirical_output_bias(
    cfg: Mapping[str, Any], dataset_dict: Mapping[str, Any], n_tracks: int, *, seed: int = 0
) -> None:
    """Replace the declarative empirical-mean mode with checkpoint-safe numeric biases."""
    head_cfg = cfg["head"]
    mode = str(head_cfg.get("output_init", "default"))
    allowed = {"default", "empirical_mean_bias", "empirical_mean_constant"}
    if mode not in allowed:
        raise ValueError(f"head.output_init must be one of {sorted(allowed)}, got {mode!r}")
    if mode == "default":
        return
    values = _empirical_track_output_bias(
        dataset_dict["train"],
        n_tracks=n_tracks,
        activation_type=str(cfg["model"].get("activation_type", "softplus")),
        max_samples=int(head_cfg.get("output_init_samples", 256)),
        seed=seed,
    )
    head_cfg["resolved_output_bias"] = values
    head_cfg["zero_output_weights"] = mode == "empirical_mean_constant"
    logger.info(
        "Initialized transfer-head output bias from %d transformed training-label samples",
        int(head_cfg.get("output_init_samples", 256)),
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
                {
                    "params": decay_params,
                    "lr": group_lr,
                    "weight_decay": trainer_cfg.weight_decay,
                    "name": family,
                }
            )
        if no_decay_params:
            param_groups.append(
                {"params": no_decay_params, "lr": group_lr, "weight_decay": 0.0, "name": family}
            )
    if not param_groups:
        param_groups.append(
            {
                "params": [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ],
                "lr": lr,
                "weight_decay": trainer_cfg.weight_decay,
                "name": "head",
            }
        )
    return AdamW(param_groups)


@dataclasses.dataclass(frozen=True)
class TrainingSchedule:
    """Authoritative dataloader, optimizer, evaluation, and checkpoint schedule."""

    steps_per_epoch: int
    total_steps: int
    explicit_max_steps: int | None
    micro_batches_per_rank: int
    effective_global_batch: int
    dropped_rows_per_epoch: int
    warmup_steps: int
    warmup_fraction: float
    log_steps: int
    eval_steps: int | None
    checkpoint_steps: int | None
    logging_events: tuple[int, ...]
    evaluation_events: tuple[int, ...]
    checkpoint_events: tuple[int, ...]


def _resolve_training_schedule(
    trainer_cfg: TrainerConfig,
    *,
    train_rows: int,
    world_size: int,
    has_eval: bool,
) -> TrainingSchedule:
    """Resolve the exact schedule used by Trainer and Accelerate.

    Training uses ``dataloader_drop_last=True``. Accelerate shards the resulting
    full batches evenly across ranks, so each rank sees
    ``train_rows // (batch_size * world_size)`` micro-batches. Trainer performs
    one final optimizer update for a partial gradient-accumulation group.

    Raises
    ------
    ValueError
        If the split is smaller than one global batch.
    """
    positive = {
        "trainer.batch_size": trainer_cfg.batch_size,
        "trainer.eval_batch_size": trainer_cfg.resolved_eval_batch_size(),
        "trainer.gradient_accumulation_steps": trainer_cfg.gradient_accumulation_steps,
        "trainer.max_epochs": trainer_cfg.max_epochs,
        "trainer.log_every_n_steps": trainer_cfg.log_every_n_steps,
        "world_size": world_size,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    for name, value in (
        ("trainer.max_steps", trainer_cfg.max_steps),
        ("trainer.eval_every_n_steps", trainer_cfg.eval_every_n_steps),
        ("trainer.evals_per_epoch", trainer_cfg.evals_per_epoch),
        ("trainer.checkpoint_every_n_steps", trainer_cfg.checkpoint_every_n_steps),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when set, got {value}")
    if train_rows <= 0:
        raise ValueError(f"Training split must contain rows, got {train_rows}")

    global_micro_batch = trainer_cfg.batch_size * world_size
    micro_batches_per_rank = train_rows // global_micro_batch
    if micro_batches_per_rank == 0:
        raise ValueError(
            f"Training split has {train_rows} rows, fewer than one global batch "
            f"({global_micro_batch} = batch_size x world_size)"
        )
    accumulation = trainer_cfg.gradient_accumulation_steps
    steps_per_epoch = math.ceil(micro_batches_per_rank / accumulation)
    total_steps = (
        trainer_cfg.max_steps
        if trainer_cfg.max_steps is not None
        else math.ceil(trainer_cfg.max_epochs * steps_per_epoch)
    )
    if trainer_cfg.warmup_steps < 0:
        raise ValueError(
            f"trainer.warmup_steps must be non-negative, got {trainer_cfg.warmup_steps}"
        )
    if trainer_cfg.warmup_steps > total_steps:
        raise ValueError(
            f"trainer.warmup_steps ({trainer_cfg.warmup_steps}) exceeds the "
            f"training budget ({total_steps} optimizer steps)"
        )

    eval_steps: int | None = None
    if has_eval:
        if trainer_cfg.eval_every_n_steps is not None:
            eval_steps = trainer_cfg.eval_every_n_steps
        elif trainer_cfg.evals_per_epoch is not None:
            eval_steps = max(round(steps_per_epoch / trainer_cfg.evals_per_epoch), 1)
        elif trainer_cfg.max_steps is None:
            eval_steps = steps_per_epoch
        else:
            eval_steps = trainer_cfg.checkpoint_every_n_steps or trainer_cfg.log_every_n_steps

    checkpoint_steps = trainer_cfg.checkpoint_every_n_steps
    if has_eval and checkpoint_steps is None and trainer_cfg.evals_per_epoch is not None:
        checkpoint_steps = eval_steps
    if has_eval and checkpoint_steps is not None and eval_steps is not None:
        if checkpoint_steps % eval_steps:
            raise ValueError(
                f"trainer.checkpoint_every_n_steps ({checkpoint_steps}) must be a multiple of "
                f"the resolved evaluation interval ({eval_steps})"
            )
        if checkpoint_steps > total_steps:
            raise ValueError(
                f"Checkpoint interval {checkpoint_steps} exceeds the training budget "
                f"of {total_steps} optimizer steps; no best-model checkpoint would exist"
            )
    if trainer_cfg.early_stopping_patience is not None and not has_eval:
        raise ValueError("trainer.early_stopping_patience requires a validation split")

    evaluation_events = (
        tuple(range(eval_steps, total_steps + 1, eval_steps)) if eval_steps is not None else ()
    )
    if has_eval and trainer_cfg.eval_on_start:
        evaluation_events = (0, *evaluation_events)
    checkpoint_events = (
        tuple(range(checkpoint_steps, total_steps + 1, checkpoint_steps))
        if checkpoint_steps is not None
        else ()
    )
    logging_events = tuple(
        range(trainer_cfg.log_every_n_steps, total_steps + 1, trainer_cfg.log_every_n_steps)
    )
    return TrainingSchedule(
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        explicit_max_steps=trainer_cfg.max_steps,
        micro_batches_per_rank=micro_batches_per_rank,
        effective_global_batch=global_micro_batch * accumulation,
        dropped_rows_per_epoch=train_rows - micro_batches_per_rank * global_micro_batch,
        warmup_steps=trainer_cfg.warmup_steps,
        warmup_fraction=trainer_cfg.warmup_steps / total_steps,
        log_steps=trainer_cfg.log_every_n_steps,
        eval_steps=eval_steps,
        checkpoint_steps=checkpoint_steps,
        logging_events=logging_events,
        evaluation_events=evaluation_events,
        checkpoint_events=checkpoint_events,
    )


def _build_training_arguments(
    output_dir: Path,
    trainer_cfg: TrainerConfig,
    schedule: TrainingSchedule,
    *,
    has_eval: bool,
    seed: int,
) -> TrainingArguments:
    if "wandb" in trainer_cfg.report_to:
        environ["WANDB_PROJECT"] = trainer_cfg.wandb_project
        environ["WANDB_JOB_TYPE"] = trainer_cfg.wandb_job_type
        if trainer_cfg.wandb_group:
            environ["WANDB_RUN_GROUP"] = trainer_cfg.wandb_group
        if trainer_cfg.wandb_tags:
            environ["WANDB_TAGS"] = ",".join(trainer_cfg.wandb_tags)
    eval_strategy = "steps" if schedule.eval_steps is not None else "no"
    save_steps = schedule.checkpoint_steps or trainer_cfg.log_every_n_steps
    save_strategy = "steps" if schedule.checkpoint_steps is not None else "no"
    logging_strategy = "steps"
    metric_for_best_model = trainer_cfg.metric_for_best_model if has_eval else None
    load_best_model_at_end = has_eval and save_strategy != "no"

    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=trainer_cfg.wandb_run_name or environ.get("WANDB_NAME") or output_dir.name,
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
        max_steps=schedule.explicit_max_steps if schedule.explicit_max_steps is not None else -1,
        warmup_steps=trainer_cfg.warmup_steps,
        gradient_accumulation_steps=trainer_cfg.gradient_accumulation_steps,
        bf16=trainer_cfg.mixed_precision == "bf16",
        fp16=trainer_cfg.mixed_precision == "fp16",
        logging_strategy=logging_strategy,
        logging_steps=schedule.log_steps,
        eval_strategy=eval_strategy,
        eval_steps=schedule.eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_total_limit=2 if save_strategy == "steps" else None,
        # Trainer re-seeds from this at train() start (shuffle order, dropout, worker
        # augmentation); left at its default every run would share seed 42.
        seed=seed,
        gradient_checkpointing=False,
        # Metadata heads build submodules (timepoint MLP, per-field embeddings) that only
        # run when the dataset carries that field, so some trainable parameters get no
        # gradient; DDP must detect them rather than wait for their reduction.
        ddp_find_unused_parameters=True,
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
    schedule: TrainingSchedule,
) -> torch.optim.lr_scheduler.LRScheduler:
    return get_scheduler(
        trainer_cfg.scheduler,
        optimizer=optimizer,
        num_warmup_steps=trainer_cfg.warmup_steps,
        num_training_steps=schedule.total_steps,
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

    def _get_dataloader(
        self,
        dataset: Any,
        description: str,
        batch_size: int,
        sampler_fn: Callable | None = None,
        is_training: bool = False,
        dataloader_key: str | None = None,
    ) -> Any:
        """Drop incomplete training batches, but evaluate every validation row."""
        original = self.args.dataloader_drop_last
        if not is_training:
            self.args.dataloader_drop_last = False
        try:
            return super()._get_dataloader(
                dataset,
                description,
                batch_size,
                sampler_fn=sampler_fn,
                is_training=is_training,
                dataloader_key=dataloader_key,
            )
        finally:
            self.args.dataloader_drop_last = original

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
            "Training configuration is missing required section(s): " + ", ".join(missing_sections)
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


def _check_dataset_layout(data_path: Path) -> None:
    """Fail fast when a directory isn't a built HF-layout Parquet dataset."""
    has_shards = any((data_path / "data").glob("*.parquet"))
    if not (data_path / "README.md").is_file() or not has_shards:
        raise FileNotFoundError(
            f"{data_path} is missing README.md or data/*.parquet shards; "
            "rebuild the dataset with `regulonado dataset`."
        )


def _load_training_dataset(data_path: Path, *, rank: int) -> dict[str, WindowParquetDataset]:
    """Build a ``WindowParquetDataset`` per split present under ``data_path``."""
    _check_dataset_layout(data_path)
    logger.info(f"[rank {rank}] loading dataset from {data_path} ...")
    t0 = perf_counter()
    dataset_dict: dict[str, WindowParquetDataset] = {}
    for split in ("train", "validation", "test"):
        if any((data_path / "data").glob(f"{split}-*.parquet")):
            dataset_dict[split] = WindowParquetDataset(data_path, split)
    if "train" not in dataset_dict:
        raise ValueError("Training dataset does not contain a 'train' split")
    sizes = {split: len(dataset) for split, dataset in dataset_dict.items()}
    logger.info(f"[rank {rank}] dataset loaded in {perf_counter() - t0:.1f}s | splits={sizes}")
    return dataset_dict


def preflight_training_schedule(cfg: Mapping[str, Any], *, world_size: int = 1) -> TrainingSchedule:
    """Resolve a dataset-backed schedule without constructing or downloading a model."""
    trainer_cfg = _resolve_trainer_config(cfg)
    data_path = Path(str(cfg["data"]["path"]))
    dataset_dict = _load_training_dataset(data_path, rank=0)
    _validate_dataset_schema(dataset_dict)
    schedule = _resolve_training_schedule(
        trainer_cfg,
        train_rows=len(dataset_dict["train"]),
        world_size=world_size,
        has_eval="validation" in dataset_dict,
    )
    _log_training_schedule(schedule, trainer_cfg, rank=0)
    return schedule


def _validate_dataset_schema(dataset_dict: Mapping[str, Any]) -> None:
    """Fail before worker startup when a dataset's stored columns lack model inputs."""
    required = {"sequence_tokens", "signal"}
    if "train" not in dataset_dict:
        raise ValueError("Training dataset does not contain a 'train' split")
    for split, dataset in dataset_dict.items():
        columns = set(dataset.schema.names)
        missing = sorted(required - columns)
        if missing:
            raise ValueError(
                f"Dataset split {split!r} is missing required column(s): "
                f"{', '.join(missing)}; available columns: {sorted(columns)}. "
                "Verify the source dataset schema before training."
            )


def _load_metadata_and_records(
    data_path: Path,
    metadata_path: Path | None,
    dataset_dict: Mapping[str, Any],
    *,
    rank: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load track metadata/records and cross-check them against the Parquet schema.

    Raises
    ------
    ValueError
        If the stored signal column's track count disagrees with the included track count.
    """
    metadata = load_dataset_metadata(data_path, metadata_path)
    records = track_records(metadata)
    logger.info(f"[rank {rank}] metadata loaded | {len(records)} tracks")

    arrow_n_tracks = dataset_dict["train"].schema.field("signal").type.list_size
    if arrow_n_tracks != len(records):
        raise ValueError(
            f"Parquet signal width ({arrow_n_tracks}) != included track count "
            f"({len(records)}); tracks.parquet and the Parquet dataset are out of sync — "
            "rebuild the dataset."
        )
    return metadata, records


def _prepare_dataset_splits(
    dataset_dict: Mapping[str, Any],
    trainer_cfg: TrainerConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    """Apply the seeded validation-sample cap, after transforms are attached."""
    dataset_dict = dict(dataset_dict)
    max_eval_samples = (
        int(trainer_cfg.max_eval_samples) if trainer_cfg.max_eval_samples is not None else None
    )
    if max_eval_samples is not None and "validation" in dataset_dict:
        val = dataset_dict["validation"]
        n_val = len(val)
        if n_val > max_eval_samples:
            rng = np.random.default_rng(seed)
            indices = sorted(rng.choice(n_val, size=max_eval_samples, replace=False).tolist())
            dataset_dict["validation"] = Subset(val, indices)
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
    model: torch.nn.Module | None = None,
) -> tuple[
    Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    np.ndarray,
    np.ndarray,
]:
    """Build the batch collate function and loss function, plus scale/background arrays.

    The scale factors and background arrays are also needed later by the eval-plot
    callback, so they are returned alongside the loss function rather than recomputed.

    When ``cfg["loss"]["learn_track_weights"]`` is set, a learnable per-track
    log-variance parameter (Kendall et al. uncertainty weighting) is registered on
    ``model`` as ``track_loss_log_var`` — assigning an ``nn.Parameter`` attribute on an
    ``nn.Module`` auto-registers it, so it is picked up by the trainer's optimizer
    without any other wiring.
    """
    track_metadata_tensors = (
        constant_track_metadata(
            records,
            condition_source=str(cfg["model"].get("condition_source", "condition_id")),
        )
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
    track_log_var: torch.nn.Parameter | None = None
    if bool(cfg["loss"].get("learn_track_weights", False)):
        if model is None:
            raise ValueError("loss.learn_track_weights requires a model to attach the parameter to")
        track_log_var = torch.nn.Parameter(torch.zeros(len(records)))
        model.track_loss_log_var = track_log_var
    trainer_cfg_for_loss = cfg.get("trainer", {})
    loss_fn = _build_loss_fn(
        cfg["loss"],
        scale_factors=scale_factors,
        clip_hard=clip_hard,
        labels_already_scaled=labels_already_scaled,
        contrast_weights=_contrast_weights_from_records(records),
        track_log_var=track_log_var,
        contrast_region_bins=int(trainer_cfg_for_loss.get("contrast_region_bins", 16)),
        contrast_pseudocount=float(trainer_cfg_for_loss.get("contrast_pseudocount", 0.1)),
        contrast_active_fraction=float(trainer_cfg_for_loss.get("contrast_active_fraction", 0.1)),
    )
    return collate_fn, loss_fn, scale_factors, background


def _prepare_model_for_training(
    model: RegulonadoModel,
    trainer_cfg: TrainerConfig,
    *,
    rank: int,
    dataset_track_names: Sequence[str] | None = None,
) -> None:
    """Apply the freeze policy and, if configured, warm-start weights from a checkpoint."""
    _apply_freeze_policy(model, trainer_cfg)
    if dataset_track_names is not None:
        assert_track_names_match(getattr(model.config, "track_names", []), dataset_track_names)
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
    schedule: TrainingSchedule,
    *,
    has_eval: bool,
    output_dir: Path,
    seed: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, TrainingArguments]:
    """Build the optimizer, LR scheduler, and HF ``TrainingArguments`` together."""
    optimizer = _build_optimizer(model, trainer_cfg)
    scheduler = _build_scheduler_for_trainer(optimizer, trainer_cfg, schedule)
    training_args = _build_training_arguments(
        output_dir,
        trainer_cfg,
        schedule,
        has_eval=has_eval,
        seed=seed,
    )
    return optimizer, scheduler, training_args


def _log_training_schedule(
    schedule: TrainingSchedule,
    trainer_cfg: TrainerConfig,
    *,
    rank: int,
) -> None:
    """Log the resolved units and event cadence before expensive model construction."""
    mode = (
        f"max_steps={schedule.explicit_max_steps}"
        if schedule.explicit_max_steps is not None
        else f"max_epochs={trainer_cfg.max_epochs}"
    )
    logger.info(
        "[rank %d] training schedule | %s | effective_global_batch=%d | "
        "micro_batches/rank/epoch=%d | optimizer_steps/epoch=%d | total_steps=%d | "
        "dropped_rows/epoch=%d | warmup=%d (%.1f%%) | log_interval=%d events=%s | "
        "eval_interval=%s events=%s | "
        "checkpoint_interval=%s events=%s",
        rank,
        mode,
        schedule.effective_global_batch,
        schedule.micro_batches_per_rank,
        schedule.steps_per_epoch,
        schedule.total_steps,
        schedule.dropped_rows_per_epoch,
        schedule.warmup_steps,
        100.0 * schedule.warmup_fraction,
        schedule.log_steps,
        schedule.logging_events,
        schedule.eval_steps,
        schedule.evaluation_events,
        schedule.checkpoint_steps,
        schedule.checkpoint_events,
    )


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
    """Assemble W&B, LR logging, early stopping, and diagnostic callbacks."""
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
            EvalExampleDiagnostics(
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
    schedule: TrainingSchedule,
    *,
    rank: int,
) -> None:
    """Log start-of-training context and run ``trainer.train()``."""
    epochs = schedule.total_steps / schedule.steps_per_epoch
    logger.info(
        f"[rank {rank}] starting trainer.train() | "
        f"total_steps={schedule.total_steps} steps_per_epoch={schedule.steps_per_epoch} "
        f"epochs={epochs:.3f} "
        f"batch_size={trainer_cfg.batch_size} grad_accum={trainer_cfg.gradient_accumulation_steps} "
        f"resume={trainer_cfg.resume_from_checkpoint} eval_on_start={trainer_cfg.eval_on_start}"
    )
    t0 = perf_counter()
    trainer.train(resume_from_checkpoint=trainer_cfg.resume_from_checkpoint)
    logger.info(f"[rank {rank}] trainer.train() returned after {perf_counter() - t0:.1f}s")


def _evaluate_test_split(
    trainer: "RegulonadoTrainer",
    test_dataset: Any,
    *,
    rank: int,
) -> dict[str, float] | None:
    """Score the final (best, when selection is on) model once on the untouched test split.

    Uses ``predict`` rather than ``evaluate`` so the one-off result lands in the run
    summary instead of step-series panels, and so it cannot reach early stopping or
    best-checkpoint selection.
    """
    if test_dataset is None:
        return None
    t0 = perf_counter()
    metrics = trainer.predict(test_dataset, metric_key_prefix="test").metrics
    logger.info(
        f"[rank {rank}] test split scored in {perf_counter() - t0:.1f}s | rows={len(test_dataset)}"
    )
    return {key: float(value) for key, value in metrics.items()}


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
    test_metrics: Mapping[str, float] | None,
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
        "test_metrics": dict(test_metrics) if test_metrics is not None else None,
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

    Loads the HF-layout Parquet dataset as a ``WindowParquetDataset`` per split,
    applies transforms, builds a model, constructs optimizer and scheduler, and
    trains using HuggingFace Trainer. Logs progress, saves checkpoints, and
    returns training history.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        Complete training configuration with keys:
        - data.path: dataset directory (built by `regulonado dataset`)
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
    logger.info(
        f"[rank {rank}] run_training start | backbone={cfg['backbone'].get('name')} "
        f"head={cfg['head'].get('type')} loss={cfg['loss'].get('name')} "
        f"output_dir={cfg.get('output_dir')}"
    )

    dataset_dict = _load_training_dataset(data_path, rank=rank)
    _validate_dataset_schema(dataset_dict)

    schedule = _resolve_training_schedule(
        trainer_cfg,
        train_rows=len(dataset_dict["train"]),
        world_size=_world_size(),
        has_eval="validation" in dataset_dict,
    )
    _log_training_schedule(schedule, trainer_cfg, rank=rank)

    metadata_path_value = cfg["data"].get("metadata_path")
    metadata_path = Path(str(metadata_path_value)) if metadata_path_value else None
    metadata, records = _load_metadata_and_records(
        data_path, metadata_path, dataset_dict, rank=rank
    )

    dataset_dict = _apply_dataset_transforms(dataset_dict, metadata, records, cfg["data"])
    logger.info(f"[rank {rank}] dataset transforms applied")

    dataset_dict = _prepare_dataset_splits(dataset_dict, trainer_cfg, seed=seed)

    _resolve_empirical_output_bias(cfg, dataset_dict, len(records), seed=seed)

    model = _build_model_with_logging(cfg, metadata, records, adapter_builder, rank=rank)
    collate_fn, loss_fn, scale_factors, background = _build_collate_and_loss(cfg, records, model)

    output_dir = Path(str(cfg.get("output_dir") or Path.cwd() / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_track_names = [
        str(r.get("track_name") or Path(r["bigwig_path"]).stem)
        for r in records
        if r.get("track_name") or r.get("bigwig_path")
    ]
    _prepare_model_for_training(
        model, trainer_cfg, rank=rank, dataset_track_names=dataset_track_names
    )

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
        model,
        trainer_cfg,
        schedule,
        has_eval="validation" in dataset_dict,
        output_dir=output_dir,
        seed=seed,
    )
    val_dataset = dataset_dict.get("validation")
    per_track_report = PerTrackMetricsReport(output_dir=output_dir, records=records)
    callbacks = [
        *_build_training_callbacks(
            cfg,
            trainer_cfg,
            val_dataset,
            collate_fn,
            scale_factors,
            background,
            records,
            output_dir,
        ),
        per_track_report,
    ]
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
        compute_metrics=make_compute_metrics(
            len(records),
            trainer_cfg.calibration_shape_pearson_weight,
            per_track_sink=per_track_report.record,
        ),
        preprocess_logits_for_metrics=make_preprocess_logits_for_metrics(
            topk_bins,
            contrast_group_weights=_contrast_weights_from_records(records),
            contrast_region_bins=trainer_cfg.contrast_region_bins,
            contrast_pseudocount=trainer_cfg.contrast_pseudocount,
            contrast_active_fraction=trainer_cfg.contrast_active_fraction,
        ),
    )

    _run_training_loop(trainer, trainer_cfg, schedule, rank=rank)
    test_metrics = _evaluate_test_split(trainer, dataset_dict.get("test"), rank=rank)
    history = _finalize_trainer_outputs(trainer, output_dir)

    return _build_training_summary(
        cfg, output_dir, seed, metadata_path, records, trainer_cfg, history, test_metrics
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
