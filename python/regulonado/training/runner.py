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
from peft import LoraModel
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
from regulonado.model.peft_adapters import attach_adapters, merge_adapters
from regulonado.target_specificity import group_index_from_records
from regulonado.training.callbacks import (
    EvalExampleDiagnostics,
    LRLogCallback,
    PerGroupMetricsReport,
    PerTrackMetricsReport,
    WandbConfigCallback,
)
from regulonado.training.config import TrainerConfig
from regulonado.training.data import WindowParquetDataset
from regulonado.training.group_contrast import (
    group_contrast_labels,
    group_replicate_weights,
    reduce_target_score,
)
from regulonado.training.label_space import (
    CountLabelSpace,
    resolve_count_label_space,
    validate_label_space,
)
from regulonado.training.losses import (
    contrast_group_weights,
    kendall_track_weighted_loss,
    log1p_huber_loss,
    mask_missing_bins,
    poisson_multinomial_binwise_loss,
    poisson_multinomial_loss,
    poisson_nll_loss,
    scaled_poisson_multinomial_loss,
    topk_additive_loss,
    topk_reweight_loss,
    track_contrast_correlation_loss,
    track_contrast_magnitude_loss,
    transfer_calibration_loss,
)
from regulonado.training.metrics import (
    make_compute_metrics,
    make_preprocess_logits_for_metrics,
)
from regulonado.training.provenance import write_provenance
from regulonado.training.specificity_panel import SpecificityPanelEvaluator

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


def group_contrast_unit_factors(
    records: Sequence[Mapping[str, Any]],
    count_space: CountLabelSpace,
    *,
    background_mode: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-track ``(k, b)`` converting ``normalised_target`` into anchor units.

    ``normalised_target = counts / exposure`` is *not* anchor units in general (only when
    ``exposure`` happens to equal the track's anchor reference with zero background — see
    the group-contrast-head plan). The exact conversion, derived from
    ``target_specificity.to_anchor_units``'s ``a = max(y - bg, 0) * sf`` (``y`` = stored
    mean coverage, ``bg`` = ``record["background"]``, ``sf`` = ``record["scale_factor"]``,
    verified numerically on real records to satisfy ``sf == 1 / (anchor_reference - bg)``)
    and ``counts = y * count_factors``, ``normalised = counts / exposure``:

        y = normalised * exposure / count_factors
        a = max(y - bg, 0) * sf = max(normalised - b, 0) * k

    with ``k = exposure * sf / count_factors`` and ``b = bg * count_factors / exposure``.

    Rather than apply this elementwise formula directly (which would require reimplementing
    :func:`regulonado.training.group_contrast.group_contrast_labels`'s smooth-then-subtract
    order), the caller instead feeds ``group_contrast_labels`` the rescaled tensor
    ``normalised_target * k`` and this function's ``k``-scaled background threshold
    ``k * b = sf * bg`` (the algebra: since ``k > 0``, ``max(x, 0) * k == max(x * k, 0)``,
    so ``max(smoothed - b, 0) * k == max(smoothed * k - k * b, 0)`` — smoothing is linear
    and commutes with the constant per-track ``k`` multiply, so it does not matter whether
    ``k`` is applied before or after :func:`~regulonado.training.group_contrast.smooth_bins`).
    This reproduces ``group_contrast_labels``' existing (already-tested) subtract-then-clamp
    step exactly, with no duplicated clamp/subtract logic.

    Returns ``(k, k * b)`` for ``background_mode="subtract"`` (the second element is what
    the caller passes as ``group_contrast_labels(..., background=...)``), or ``(k, None)``
    for ``"scale-only"`` (matching :func:`~regulonado.target_specificity.to_anchor_units`'s
    ``background_mode="scale-only"``, which skips subtraction entirely: ``a = normalised * k
    == y * sf``).
    """
    if background_mode not in ("subtract", "scale-only"):
        raise ValueError(
            "trainer.group_contrast_background must be 'subtract' or 'scale-only', got "
            f"{background_mode!r}"
        )
    scale_factors, _, _, background = resolve_scale_and_clip(records)
    exposure = count_space.exposure.astype(np.float64)
    count_factors = count_space.count_factors.astype(np.float64)
    sf = scale_factors.astype(np.float64)
    k = (exposure * sf / count_factors).astype(np.float32)
    if background_mode == "scale-only":
        return k, None
    scaled_background = (sf * background.astype(np.float64)).astype(np.float32)
    return k, scaled_background


@dataclasses.dataclass(frozen=True)
class GroupContrastLossConfig:
    """Everything :func:`_build_loss_fn` needs to compute the two group-contrast terms."""

    group_weights: torch.Tensor  # [G, T], from group_replicate_weights
    unit_k: torch.Tensor  # [T]
    unit_b: torch.Tensor | None  # [T], only for background_mode="subtract"
    target_index: int | None  # into group axis; required when score_weight > 0
    smoothing_bins: int
    pseudocount: float
    gauge: str
    quantile: float
    floor: float
    clamp: tuple[float, float]
    channel_weight: float
    score_weight: float
    delta: float


def _resolve_group_contrast_loss_config(
    loss_cfg: Mapping[str, Any],
    trainer_cfg: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    count_space: CountLabelSpace | None,
    group_contrast_n_groups: int,
    group_contrast_group_names: Sequence[str],
    require_active_weight: bool = True,
) -> GroupContrastLossConfig | None:
    """Build the group-contrast loss config, or ``None`` when both weights are unset/0.

    Raises ``ValueError`` (at build time, before training starts) when either weight is
    enabled without the group head, without ``data.exposure`` (count label space), or when
    ``trainer.group_contrast_target`` is missing/unresolvable and ``group_score_weight`` is
    set.

    ``require_active_weight=False`` (used to resolve the same config for group-space
    *metrics*, independent of whether the group-contrast loss terms are actually weighted)
    skips the "both weights are 0 -> None" short-circuit and, when the group head or
    ``data.exposure`` are missing, returns ``None`` instead of raising — metrics are simply
    unavailable rather than a hard training-config error. An explicitly misconfigured
    ``trainer.group_contrast_target`` still raises either way.
    """
    channel_weight = float(loss_cfg.get("group_contrast_weight") or 0.0)
    score_weight = float(loss_cfg.get("group_score_weight") or 0.0)
    if require_active_weight and channel_weight <= 0.0 and score_weight <= 0.0:
        return None
    if group_contrast_n_groups <= 0:
        if not require_active_weight:
            return None
        raise ValueError(
            "loss.group_contrast_weight and loss.group_score_weight require "
            "head.group_contrast_enabled (no group-contrast channels on the model)"
        )
    if count_space is None:
        if not require_active_weight:
            return None
        raise ValueError(
            "loss.group_contrast_weight and loss.group_score_weight require "
            "data.label_space=counts with a data.exposure setting (e.g. 'anchor'); "
            "got data.label_space=transformed, so anchor units cannot be recovered"
        )
    target_name = trainer_cfg.get("group_contrast_target")
    target_index: int | None = None
    if target_name is not None:
        group_names = list(group_contrast_group_names)
        if target_name not in group_names:
            raise ValueError(
                f"trainer.group_contrast_target {target_name!r} not found in the model's "
                f"group_contrast_group_names: {group_names}"
            )
        target_index = group_names.index(target_name)
    if score_weight > 0.0 and target_index is None:
        raise ValueError(
            "loss.group_score_weight requires trainer.group_contrast_target to name one of "
            f"the model's group_contrast_group_names: {list(group_contrast_group_names)}"
        )

    background_mode = str(trainer_cfg.get("group_contrast_background", "subtract"))
    unit_k, unit_b = group_contrast_unit_factors(
        records, count_space, background_mode=background_mode
    )
    group_index, _ = group_index_from_records(records)
    group_weights = group_replicate_weights(
        torch.as_tensor(group_index), group_contrast_n_groups
    )
    clamp_min = float(trainer_cfg.get("group_contrast_clamp_min", -6.0))
    clamp_max = float(trainer_cfg.get("group_contrast_clamp_max", 6.0))
    return GroupContrastLossConfig(
        group_weights=group_weights,
        unit_k=torch.as_tensor(unit_k),
        unit_b=None if unit_b is None else torch.as_tensor(unit_b),
        target_index=target_index,
        smoothing_bins=int(trainer_cfg.get("group_contrast_smoothing_bins", 31)),
        pseudocount=float(trainer_cfg.get("group_contrast_pseudocount", 0.1)),
        gauge=str(trainer_cfg.get("group_contrast_gauge", "median")),
        quantile=float(trainer_cfg.get("group_contrast_quantile", 0.9)),
        floor=float(trainer_cfg.get("group_contrast_floor", 0.139)),
        clamp=(clamp_min, clamp_max),
        channel_weight=channel_weight,
        score_weight=score_weight,
        delta=float(loss_cfg.get("group_contrast_delta") or 1.0),
    )


def _build_specificity_panel(
    trainer_cfg: TrainerConfig,
    metadata: Mapping[str, Any],
    *,
    group_contrast_group_names: Sequence[str],
) -> SpecificityPanelEvaluator | None:
    """Build the eval-time specificity panel, or ``None`` when no panel is configured.

    Raises ``ValueError`` before training starts when the panel is set without a FASTA, a
    group-contrast head, or a ``trainer.group_contrast_target`` in the model's groups.
    """
    if not trainer_cfg.specificity_panel_path:
        return None
    if not trainer_cfg.specificity_panel_fasta:
        raise ValueError("trainer.specificity_panel_path requires trainer.specificity_panel_fasta")
    group_names = list(group_contrast_group_names)
    if not group_names:
        raise ValueError(
            "trainer.specificity_panel_path requires head.group_contrast_enabled "
            "(the panel scores the group-contrast channels)"
        )
    target = trainer_cfg.group_contrast_target
    if target not in group_names:
        raise ValueError(
            f"trainer.specificity_panel_path requires trainer.group_contrast_target to name one "
            f"of the model's group_contrast_group_names; got {target!r}"
        )
    return SpecificityPanelEvaluator(
        trainer_cfg.specificity_panel_path,
        trainer_cfg.specificity_panel_fasta,
        target_index=group_names.index(target),
        quantile=trainer_cfg.group_contrast_quantile,
        n_pred_bins=int(metadata.get("n_pred_bins", 6_144)),
        bin_size=int(metadata.get("bin_size", 32)),
        context_length=int(metadata.get("context_length", 524_288)),
        batch_size=trainer_cfg.specificity_panel_batch_size,
    )


def _group_contrast_metrics_kwargs(cfg: GroupContrastLossConfig | None) -> dict[str, Any]:
    """``make_preprocess_logits_for_metrics``'s ``group_contrast_*`` kwargs from ``cfg``.

    ``cfg=None`` (no group head, or one not resolvable into a metrics config) returns
    ``group_contrast_group_weights=None`` and defaults for the rest — the preprocess
    closure ignores everything except ``group_weights`` when it is ``None``, so the exact
    defaults used here for the other fields don't matter in that case.
    """
    if cfg is None:
        return {
            "group_contrast_group_weights": None,
            "group_contrast_unit_k": None,
            "group_contrast_unit_b": None,
            "group_contrast_target_index": None,
            "group_contrast_smoothing_bins": 31,
            "group_contrast_pseudocount": 0.1,
            "group_contrast_gauge": "median",
            "group_contrast_floor": 0.139,
            "group_contrast_clamp": (-6.0, 6.0),
            "group_contrast_quantile": 0.9,
        }
    return {
        "group_contrast_group_weights": cfg.group_weights,
        "group_contrast_unit_k": cfg.unit_k,
        "group_contrast_unit_b": cfg.unit_b,
        "group_contrast_target_index": cfg.target_index,
        "group_contrast_smoothing_bins": cfg.smoothing_bins,
        "group_contrast_pseudocount": cfg.pseudocount,
        "group_contrast_gauge": cfg.gauge,
        "group_contrast_floor": cfg.floor,
        "group_contrast_clamp": cfg.clamp,
        "group_contrast_quantile": cfg.quantile,
    }


def _group_contrast_loss_terms(
    group_pred: torch.Tensor,
    normalised_target: torch.Tensor,
    cfg: GroupContrastLossConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(channel_loss, score_loss)`` for the group-contrast head; each a scalar tensor.

    Both terms keep a gradient path (``+ group_pred.sum() * 0.0``) even when the mask
    excludes every bin, matching :func:`track_contrast_magnitude_loss`'s convention.
    """
    unit_k = cfg.unit_k.to(device=group_pred.device, dtype=group_pred.dtype)
    unit_b = (
        None
        if cfg.unit_b is None
        else cfg.unit_b.to(device=group_pred.device, dtype=group_pred.dtype)
    )
    scaled_target = normalised_target * unit_k[None, :, None]
    labels, mask = group_contrast_labels(
        scaled_target,
        group_weights=cfg.group_weights,
        background=unit_b,
        smoothing_bins=cfg.smoothing_bins,
        pseudocount=cfg.pseudocount,
        gauge=cfg.gauge,  # type: ignore[arg-type]
        floor=cfg.floor,
        clamp=cfg.clamp,
    )
    zero_keepalive = group_pred.sum() * 0.0
    channel_loss = zero_keepalive
    if cfg.channel_weight > 0.0:
        huber = torch.nn.functional.huber_loss(
            group_pred, labels, delta=cfg.delta, reduction="none"
        )
        denom = (mask.sum() * group_pred.shape[1]).clamp_min(1.0)
        channel_loss = (huber * mask).sum() / denom + zero_keepalive
    score_loss = zero_keepalive
    if cfg.score_weight > 0.0:
        assert cfg.target_index is not None
        pred_score = reduce_target_score(
            group_pred, target_index=cfg.target_index, quantile=cfg.quantile
        )
        label_score = reduce_target_score(
            labels, target_index=cfg.target_index, quantile=cfg.quantile
        )
        mask_flat = mask[:, 0, :]
        huber_s = torch.nn.functional.huber_loss(
            pred_score, label_score, delta=cfg.delta, reduction="none"
        )
        denom_s = mask_flat.sum().clamp_min(1.0)
        score_loss = (huber_s * mask_flat).sum() / denom_s + zero_keepalive
    return channel_loss, score_loss


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
    exposure: torch.Tensor | None = None,
    group_contrast: GroupContrastLossConfig | None = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]:
    """Build the configured base loss, plus the cross-track contrast term when weighted.

    Missing bins (NaN targets) are removed from every term: prediction and target are
    both zeroed there, which contributes nothing to the per-bin terms and no gradient.

    With ``exposure`` (``data.label_space: counts``), targets are counts and predictions
    are exposure-normalised rates: the base loss sees ``pred * exposure`` against the
    counts, and the contrast terms compare ``pred`` with ``target / exposure``.

    The returned closure takes an optional third argument, ``group_pred`` — the
    group-contrast head's ``[B, G, L]`` channels, sliced from the model's concatenated
    logits by the caller. When ``group_contrast`` (this function's argument) is ``None``,
    ``group_pred`` is ignored entirely and behaviour is byte-identical to a build without
    group-contrast support. The two group-contrast term values are exposed for logging as
    ``loss_fn.last_group_channel_loss`` / ``loss_fn.last_group_score_loss`` (Python floats,
    updated on every call that supplies ``group_pred``; ``None`` otherwise).
    """
    if exposure is not None and str(loss_cfg.get("name")) == "scaled_poisson_multinomial":
        raise ValueError(
            "loss.name=scaled_poisson_multinomial applies its own scaling; use "
            "poisson_multinomial or poisson_multinomial_binwise with data.label_space=counts"
        )
    base_loss = _build_base_loss_fn(
        loss_cfg,
        scale_factors=scale_factors,
        clip_hard=clip_hard,
        labels_already_scaled=labels_already_scaled,
        track_log_var=track_log_var,
    )
    contrast_weight = float(loss_cfg.get("contrast_weight") or 0.0)
    magnitude_weight = float(loss_cfg.get("contrast_magnitude_weight") or 0.0)
    use_contrast = contrast_weight > 0.0 or magnitude_weight > 0.0
    if use_contrast and (contrast_weights is None or contrast_weights.shape[0] == 0):
        raise ValueError(
            "loss.contrast_weight and loss.contrast_magnitude_weight require tracks labelled "
            "with assay_class and group, with at least two groups sharing one assay_class"
        )
    geometry = dict(
        region_bins=contrast_region_bins,
        pseudocount=contrast_pseudocount,
        active_fraction=contrast_active_fraction,
    )

    def loss_fn(
        pred: torch.Tensor, target: torch.Tensor, group_pred: torch.Tensor | None = None
    ) -> torch.Tensor:
        pred, target = mask_missing_bins(pred, target)
        if exposure is None:
            base_pred, normalised_target = pred, target
        else:
            track_exposure = exposure.to(device=pred.device, dtype=pred.dtype)[:, None]
            base_pred, normalised_target = pred * track_exposure, target / track_exposure
        loss = base_loss(base_pred, target)
        if contrast_weight > 0.0:
            loss = loss + contrast_weight * track_contrast_correlation_loss(
                pred, normalised_target, contrast_weights, **geometry
            )
        if magnitude_weight > 0.0:
            loss = loss + magnitude_weight * track_contrast_magnitude_loss(
                pred, normalised_target, contrast_weights, **geometry
            )
        if group_contrast is not None and group_pred is not None:
            channel_loss, score_loss = _group_contrast_loss_terms(
                group_pred, normalised_target, group_contrast
            )
            loss_fn.last_group_channel_loss = float(channel_loss.detach())
            loss_fn.last_group_score_loss = float(score_loss.detach())
            if group_contrast.channel_weight > 0.0:
                loss = loss + group_contrast.channel_weight * channel_loss
            if group_contrast.score_weight > 0.0:
                loss = loss + group_contrast.score_weight * score_loss
        return loss

    loss_fn.last_group_channel_loss = None
    loss_fn.last_group_score_loss = None
    return loss_fn


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


def _resolve_count_label_space(
    data_cfg: Mapping[str, Any],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    rank: int = 0,
) -> CountLabelSpace | None:
    """Per-track count factors and exposure for ``data.label_space: counts``, else None."""
    label_space = str(data_cfg.get("label_space", "transformed"))
    count_unit = str(data_cfg.get("count_unit", "fragments"))
    exposure = str(data_cfg.get("exposure", "anchor"))
    validate_label_space(label_space, count_unit, exposure)
    if label_space != "counts":
        return None
    count_space = resolve_count_label_space(
        records,
        count_unit=count_unit,
        exposure=exposure,
        bin_size=int(metadata.get("bin_size", 32)),
    )
    if bool(data_cfg.get("enable_rc_aug", False)):
        # RC augmentation swaps paired strand channels' labels, but each output channel
        # keeps its own exposure, so pairs must share one for the offset to stay exact.
        rc_perm = build_rc_permutation(list(records))
        if rc_perm is not None:
            ratio = count_space.exposure / count_space.exposure[rc_perm]
            if np.any(np.abs(np.log(ratio)) > np.log(1.05)):
                logger.warning(
                    "RC augmentation pairs strand tracks whose exposures differ by more than "
                    "5%%; swapped labels are fitted against the other strand's exposure"
                )
    logger.info(
        "[rank %d] label_space=counts count_unit=%s exposure=%s | count factor median=%.4g, "
        "exposure median=%.4g (range %.4g-%.4g) count units per output unit",
        rank,
        count_unit,
        exposure,
        float(np.median(count_space.count_factors)),
        float(np.median(count_space.exposure)),
        float(count_space.exposure.min()),
        float(count_space.exposure.max()),
    )
    return count_space


def _apply_dataset_transforms(
    dataset_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    data_cfg: Mapping[str, Any],
    count_space: CountLabelSpace | None = None,
) -> Mapping[str, Any]:
    scale_factors, clip_soft, clip_hard, background = resolve_scale_and_clip(records)
    label_space_kwargs: dict[str, Any] = {
        "label_space": "counts" if count_space is not None else "transformed",
        "count_factors": None if count_space is None else count_space.count_factors,
        "mask_missing": bool(data_cfg.get("mask_missing", True)),
    }
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
        **label_space_kwargs,
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
        **label_space_kwargs,
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

    group_contrast_n_groups = 0
    group_contrast_group_names: list[str] = []
    if bool(head_cfg.get("group_contrast_enabled", False)):
        # Canonical group ordering: derived once, here, from the same records list used
        # to build the rest of the model config, so head construction and any later
        # label computation see the same group axis (see target_specificity plan).
        _, group_contrast_group_names = group_index_from_records(records)
        group_contrast_n_groups = len(group_contrast_group_names)
        if group_contrast_n_groups < 2:
            raise ValueError(
                "head.group_contrast_enabled requires at least two distinct 'group' "
                f"values across the track records; got {group_contrast_n_groups}"
            )

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
        group_contrast_n_groups=group_contrast_n_groups,
        group_contrast_hidden=int(head_cfg.get("group_contrast_hidden", 512)),
        group_contrast_mlp_hidden=(
            int(head_cfg["group_contrast_mlp_hidden"])
            if head_cfg.get("group_contrast_mlp_hidden") is not None
            else None
        ),
        group_contrast_dropout=float(head_cfg.get("group_contrast_dropout", 0.0)),
        group_contrast_group_names=group_contrast_group_names,
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
    label_divisor: np.ndarray | None = None,
) -> list[float]:
    """Estimate per-track transformed-label means from a seeded random sample of rows.

    ``label_divisor`` (the count label space's exposure) puts count labels into the
    model's output units first. Missing (NaN) bins are left out of the mean.

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
    counts = np.zeros(n_tracks, dtype=np.float64)
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
        if label_divisor is not None:
            labels = labels / np.asarray(label_divisor, dtype=np.float64).reshape(-1, 1)
        totals += np.nansum(labels, axis=1)
        counts += np.sum(~np.isnan(labels), axis=1)
    if not np.any(counts):
        raise ValueError("Cannot initialize output bias from an empty training dataset")
    return _inverse_output_activation_mean(
        totals / np.maximum(counts, 1.0), activation_type
    ).tolist()


def _resolve_empirical_output_bias(
    cfg: Mapping[str, Any],
    dataset_dict: Mapping[str, Any],
    n_tracks: int,
    *,
    seed: int = 0,
    count_space: CountLabelSpace | None = None,
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
        label_divisor=None if count_space is None else count_space.exposure,
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
    # Falls back to the head LR, not backbone_lr: adapter params live inside the
    # backbone module tree, so id()-based backbone/head classification alone would
    # sweep them into the backbone group, and presets set backbone_learning_rate as
    # low as 5e-8 — effectively not training the adapters at all. Classifying by
    # "lora_" in name below (before the trunk/head id() check) is what fixes that;
    # this fallback is what keeps an unset lora_learning_rate from reintroducing it.
    lora_lr = trainer_cfg.lora_learning_rate or lr

    head_ids = {id(parameter) for parameter in model.head_parameters()}
    trunk_ids = {id(parameter) for parameter in model.trunk_parameters()}
    no_decay_terms = ("bias", "norm", "ln", "embedding", "scale", "shift")
    grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {
        ("backbone", True): [],
        ("backbone", False): [],
        ("head", True): [],
        ("head", False): [],
        ("lora", True): [],
        ("lora", False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            family = "lora"
        elif id(parameter) in trunk_ids:
            family = "backbone"
        elif id(parameter) in head_ids:
            family = "head"
        else:
            family = "head"
        use_decay = parameter.ndim > 1 and not any(term in name.lower() for term in no_decay_terms)
        grouped[(family, use_decay)].append(parameter)

    param_groups: list[dict[str, Any]] = []
    for family, group_lr in (("backbone", backbone_lr), ("head", lr), ("lora", lora_lr)):
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


def _model_track_channel_count(model: torch.nn.Module) -> int | None:
    """Per-track channel count carried on ``model`` (or a DDP-wrapped ``model.module``).

    A composite group-contrast head sets this on ``RegulonadoModel`` so logits can be
    split into their leading ``[B, T, L]`` per-track slice before comparing against
    per-track labels; a plain single-output head has no need to split anything, so this
    resolves to ``None`` for it (and for any model that never set the attribute), leaving
    logits untouched.
    """
    count = getattr(model, "track_channel_count", None)
    if count is None:
        count = getattr(getattr(model, "module", None), "track_channel_count", None)
    return count


def _split_track_and_group_logits(
    logits: torch.Tensor, model: torch.nn.Module
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split concatenated ``[B, T(+G), L]`` logits into per-track and group-contrast slices.

    Shared by ``compute_loss`` and ``prediction_step`` so the split logic (and the "labels
    are always per-track only" invariant it protects — see the group-contrast-head plan's
    silent-transpose trap) can never diverge between training and evaluation. A plain
    single-output head (``track_channel_count is None``) returns ``(logits, None)``
    unchanged.
    """
    track_channel_count = _model_track_channel_count(model)
    if track_channel_count is None:
        return logits, None
    track_logits = logits[..., :track_channel_count, :]
    group_logits = (
        logits[..., track_channel_count:, :] if logits.shape[-2] > track_channel_count else None
    )
    return track_logits, group_logits


class RegulonadoTrainer(Trainer):
    """Trainer subclass with custom loss, checkpoint saving, and metrics preprocessing.

    Saves checkpoints as self-contained HF ``PreTrainedModel`` directories (``config.json`` +
    ``model.safetensors``) so any checkpoint can be loaded with
    ``RegulonadoModel.from_pretrained(checkpoint_dir)`` without external metadata files.

    Applies ``preprocess_logits_for_metrics`` inside ``prediction_step`` (with raw labels) and
    then reduces labels to ``[B, T]`` before accumulation to avoid OOM.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._loss_fn: Callable = kwargs.pop(
            "loss_fn",
            lambda pred, target, group_pred=None: torch.nn.functional.mse_loss(pred, target),
        )
        self._metrics_preprocess: Callable | None = kwargs.pop(
            "preprocess_logits_for_metrics", None
        )
        self._specificity_panel: SpecificityPanelEvaluator | None = kwargs.pop(
            "specificity_panel", None
        )
        # Set only when trainer_cfg.adapter.enabled: the peft LoraModel handle wrapping
        # model.backbone.model, used by save_model to merge adapters into base weights
        # for the run-root checkpoint only (never intermediate checkpoint-* saves — see
        # save_model).
        self._lora_model: LoraModel | None = kwargs.pop("lora_model", None)
        self._merge_adapters_on_final_save: bool = kwargs.pop("merge_adapters_on_final_save", False)
        self._adapters_merged = False
        # Accumulates the group-contrast loss terms between log events (the same
        # since-last-log averaging HF applies to "loss" itself), so a flat group term is
        # visible in the same logs as everything else rather than only in the total loss —
        # see the group-contrast-head plan's "group term starvation" risk.
        self._group_contrast_sum = 0.0
        self._group_score_sum = 0.0
        self._group_contrast_count = 0
        super().__init__(*args, **kwargs)

    def _record_group_contrast_metrics(self) -> None:
        """Accumulate this step's group-contrast term values, if the loss computed any."""
        channel_value = getattr(self._loss_fn, "last_group_channel_loss", None)
        score_value = getattr(self._loss_fn, "last_group_score_loss", None)
        if channel_value is None and score_value is None:
            return
        self._group_contrast_sum += channel_value or 0.0
        self._group_score_sum += score_value or 0.0
        self._group_contrast_count += 1

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        """Merge the averaged-since-last-log group-contrast term values into ``logs``.

        Mirrors how Trainer's own ``loss`` entry is an average since the previous log
        event, not the last micro-step's value, so the two are directly comparable.
        Mutating ``logs`` before delegating to ``Trainer.log`` (rather than reading it
        back afterwards via an ``on_log`` callback) guarantees every configured reporting
        backend — console, W&B, etc. — sees these keys, since they all receive this same
        dict.
        """
        if self._group_contrast_count > 0 and "loss" in logs:
            logs = dict(logs)
            logs["group_channel_loss"] = self._group_contrast_sum / self._group_contrast_count
            logs["group_score_loss"] = self._group_score_sum / self._group_contrast_count
            self._group_contrast_sum = 0.0
            self._group_score_sum = 0.0
            self._group_contrast_count = 0
        super().log(logs, *args, **kwargs)

    def evaluation_loop(self, *args: Any, **kwargs: Any) -> Any:
        """Run the standard evaluation loop, then score the specificity panel when configured.

        Hooked here rather than in ``evaluate`` so the ``<prefix>_panel_*`` metrics are part
        of the loop's output before ``evaluate`` logs it and fires ``on_evaluate``: early
        stopping and ``metric_for_best_model`` can then select on them. Rank 0 predicts;
        other ranks receive its values, keeping best-model decisions in step.
        """
        output = super().evaluation_loop(*args, **kwargs)
        if self._specificity_panel is None:
            return output
        prefix = kwargs.get("metric_key_prefix", args[4] if len(args) > 4 else "eval")
        panel: list[dict[str, float] | None] = [None]
        if self.is_world_process_zero():
            panel[0] = self._specificity_panel.evaluate(self.accelerator.unwrap_model(self.model))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(panel, src=0)
        output.metrics.update({f"{prefix}_{key}": value for key, value in (panel[0] or {}).items()})
        return output

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
            # A composite group-contrast head concatenates group channels after the
            # per-track ones, so logits may be [B, T + G, L]; split to the per-track
            # slice *before* aligning against labels, which are always per-track only.
            # Comparing labels against the full (unsplit) logits here would silently
            # transpose them into [B, L, T] whenever T + G != T, corrupting training.
            track_logits, group_logits = _split_track_and_group_logits(logits, model)
            # Labels may arrive as [B, L, T]; align to [B, T, L] expected by the loss.
            aligned = (
                labels
                if labels.shape[-2:] == track_logits.shape[-2:]
                else labels.transpose(-2, -1)
            )
            loss = self._loss_fn(track_logits, aligned, group_logits)
            outputs["loss"] = loss
            self._record_group_contrast_metrics()

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
        # Merge adapters only for the run-root save, never for intermediate
        # checkpoint-* saves: HF passes the checkpoint subdirectory for periodic
        # saves and the root output_dir for the final trainer.save_model() call, so
        # comparing against self.args.output_dir distinguishes them.
        # load_best_model_at_end=True reloads a checkpoint's state dict into the
        # live, still-injected model, so a merged (unloaded) checkpoint would have
        # different key names and fail to load. Consequence: intermediate
        # checkpoint-* dirs are not loadable by inference.py — _find_weights
        # (inference.py:141-183) prefers the run-root model.safetensors (the merged
        # one), so the normal path works, but its checkpoint-* fallback would hand
        # back an un-merged (still peft-wrapped) model.
        is_root_save = output_path.resolve() == Path(self.args.output_dir).resolve()
        if (
            is_root_save
            and self._lora_model is not None
            and self._merge_adapters_on_final_save
            and not self._adapters_merged
        ):
            merge_adapters(self._lora_model)
            self._adapters_merged = True
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
        group_logits: torch.Tensor | None = None
        if logits is not None:
            # Same split as compute_loss: a composite head's logits carry group-contrast
            # channels after the per-track ones. The per-track slice is what per-track
            # metrics preprocessing (built for [B, T, L] logits) expects; the group slice
            # (if any) is passed through separately so group-space metrics can see it too.
            logits, group_logits = _split_track_and_group_logits(logits, model)
        if self._metrics_preprocess is not None and logits is not None and labels is not None:
            logits = self._metrics_preprocess(logits, labels, group_logits=group_logits)
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
    count_space: CountLabelSpace | None = None,
) -> tuple[
    Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]],
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor],
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
    model_config = getattr(model, "config", None)
    group_contrast_cfg = _resolve_group_contrast_loss_config(
        cfg["loss"],
        trainer_cfg_for_loss,
        records=records,
        count_space=count_space,
        group_contrast_n_groups=int(getattr(model_config, "group_contrast_n_groups", 0) or 0),
        group_contrast_group_names=getattr(model_config, "group_contrast_group_names", None) or [],
    )
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
        exposure=None if count_space is None else torch.as_tensor(count_space.exposure),
        group_contrast=group_contrast_cfg,
    )
    return collate_fn, loss_fn, scale_factors, background


def _prepare_model_for_training(
    model: RegulonadoModel,
    trainer_cfg: TrainerConfig,
    *,
    rank: int,
    dataset_track_names: Sequence[str] | None = None,
) -> LoraModel | None:
    """Apply the freeze policy, optionally warm-start, then optionally inject adapters.

    Order is load-bearing: freeze -> warm start -> adapter injection.
    - Injection must follow warm start: peft renames ``...conv_layer.weight`` to
      ``...conv_layer.base_layer.weight``, so injecting first would break
      warm-start state-dict key matching.
    - Injection must follow the freeze pass: peft marks only adapter parameters
      trainable, and running the freeze pass afterwards would undo that.

    Returns the injected ``LoraModel`` handle when ``trainer_cfg.adapter.enabled``,
    so callers can thread it through to the trainer for the merge-on-save step;
    ``None`` otherwise.
    """
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

    lora_model: LoraModel | None = None
    if trainer_cfg.adapter.enabled:
        lora_model = attach_adapters(model, trainer_cfg.adapter)
        n_trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"[rank {rank}] LoRA/LoCon adapters attached | "
            f"{n_trainable_after / 1e6:.1f}M trainable params (post-injection)"
        )
    return lora_model


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
    count_space: CountLabelSpace | None = None,
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
                label_divisor=None if count_space is None else count_space.exposure,
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

    count_space = _resolve_count_label_space(cfg["data"], metadata, records, rank=rank)
    dataset_dict = _apply_dataset_transforms(
        dataset_dict, metadata, records, cfg["data"], count_space
    )
    logger.info(f"[rank {rank}] dataset transforms applied")

    dataset_dict = _prepare_dataset_splits(dataset_dict, trainer_cfg, seed=seed)

    _resolve_empirical_output_bias(
        cfg, dataset_dict, len(records), seed=seed, count_space=count_space
    )

    model = _build_model_with_logging(cfg, metadata, records, adapter_builder, rank=rank)
    collate_fn, loss_fn, scale_factors, background = _build_collate_and_loss(
        cfg, records, model, count_space=count_space
    )

    output_dir = Path(str(cfg.get("output_dir") or Path.cwd() / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_track_names = [
        str(r.get("track_name") or Path(r["bigwig_path"]).stem)
        for r in records
        if r.get("track_name") or r.get("bigwig_path")
    ]
    lora_model = _prepare_model_for_training(
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
            count_space,
        ),
        per_track_report,
    ]

    # Group-space evaluation metrics reuse the same config the group-contrast loss builds
    # (group replicate weights, unit conversion, target index, ...), but resolved
    # independently of whether the loss terms are actually weighted (require_active_weight
    # =False): metrics on a configured group head should be visible even mid-warmup, before
    # loss.group_contrast_weight/loss.group_score_weight are turned on.
    model_config_for_metrics = getattr(model, "config", None)
    group_contrast_n_groups = int(
        getattr(model_config_for_metrics, "group_contrast_n_groups", 0) or 0
    )
    group_contrast_group_names = (
        getattr(model_config_for_metrics, "group_contrast_group_names", None) or []
    )
    group_contrast_metrics_cfg = _resolve_group_contrast_loss_config(
        cfg["loss"],
        cfg.get("trainer", {}),
        records=records,
        count_space=count_space,
        group_contrast_n_groups=group_contrast_n_groups,
        group_contrast_group_names=group_contrast_group_names,
        require_active_weight=False,
    )
    per_group_report: PerGroupMetricsReport | None = None
    if group_contrast_metrics_cfg is not None:
        per_group_report = PerGroupMetricsReport(
            output_dir=output_dir, group_names=group_contrast_group_names
        )
        callbacks.append(per_group_report)
    specificity_panel = _build_specificity_panel(
        trainer_cfg, metadata, group_contrast_group_names=group_contrast_group_names
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
        specificity_panel=specificity_panel,
        lora_model=lora_model,
        merge_adapters_on_final_save=trainer_cfg.adapter.merge_on_final_save,
        compute_metrics=make_compute_metrics(
            len(records),
            trainer_cfg.calibration_shape_pearson_weight,
            per_track_sink=per_track_report.record,
            per_group_sink=None if per_group_report is None else per_group_report.record,
        ),
        preprocess_logits_for_metrics=make_preprocess_logits_for_metrics(
            topk_bins,
            contrast_group_weights=_contrast_weights_from_records(records),
            contrast_region_bins=trainer_cfg.contrast_region_bins,
            contrast_pseudocount=trainer_cfg.contrast_pseudocount,
            contrast_active_fraction=trainer_cfg.contrast_active_fraction,
            label_divisor=None if count_space is None else torch.as_tensor(count_space.exposure),
            **_group_contrast_metrics_kwargs(group_contrast_metrics_cfg),
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
