"""Lean HF ``Trainer`` entry point for :class:`~regulonado.training.cached.model.RegionCountModel`.

Driven by the Hydra config ``python/configs/train_cached.yaml`` (``data.*``/``model.*``/
``loss.*``/``trainer: TrainerConfig``), the same shape as :mod:`regulonado.training.runner`
but far smaller: the backbone is already frozen and cached
(:mod:`regulonado.embeddings.cache`), so there is no backbone to build, freeze-policy, or
adapter here -- only the pooling head, and a plain ``transformers.Trainer`` is enough
(``RegionCountModel.forward`` already returns an HF-style ``ModelOutput`` with ``loss``
computed when ``labels`` is given, so no custom ``compute_loss``/``Trainer`` subclass is
needed). Reuses :mod:`regulonado.training.runner`'s schedule/``TrainingArguments``
resolution, :mod:`regulonado.training.callbacks`, and
:mod:`regulonado.training.provenance` verbatim -- see each import below.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from torch.optim import AdamW
from torch.utils.data import Subset
from transformers import EarlyStoppingCallback, Trainer, default_data_collator

from regulonado.counts.dataset import RegionCountData
from regulonado.embeddings.cache import EmbeddingManifest, EmbeddingStore
from regulonado.training.cached.data import (
    CachedRegionDataset,
    PreparedRegionData,
    RegionsDataConfig,
    attach_region_rows,
    prepare_region_data,
)
from regulonado.training.cached.metrics import GroupedCountMetrics
from regulonado.training.cached.model import RegionCountConfig, RegionCountModel
from regulonado.training.callbacks import LRLogCallback, WandbConfigCallback
from regulonado.training.config import TrainerConfig
from regulonado.training.provenance import write_provenance
from regulonado.training.runner import (
    TrainingSchedule,
    _build_scheduler_for_trainer,
    _build_training_arguments,
    _resolve_training_schedule,
    _seed_everything,
    _world_size,
)

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
# Sections read key-by-key (``cfg["data"].get(...)``), so a misspelt key would otherwise
# be silently ignored; train_cached.yaml declares every key they accept.
_CHECKED_SECTIONS = ("data", "model", "loss")


def validate_cached_config(cfg: Mapping[str, Any]) -> None:
    """Reject keys ``train_cached.yaml`` does not declare, and a ``trainer`` that
    ``TrainerConfig`` rejects.

    Raises:
        ValueError: naming every unknown ``data``/``model``/``loss`` key, or the trainer error.
    """
    base = OmegaConf.load(CONFIG_DIR / "train_cached.yaml")
    unknown = [
        f"{section}.{key}"
        for section in _CHECKED_SECTIONS
        for key in (cfg.get(section) or {})
        if key not in base[section].keys()
    ]
    if unknown:
        raise ValueError(f"unknown region-training setting(s): {', '.join(sorted(unknown))}")
    _resolve_trainer_config(cfg)


def resolved_cached_config(preset: str, overrides: list[str]) -> str:
    """Compose a ``cached_experiment`` preset over ``train_cached.yaml``, for ``--print-config``.

    Mirrors :func:`regulonado.training.compose.resolved_training_config`, kept separate
    (rather than generalising that function) since it composes a different root config
    with a different experiment-group name (``cached_experiment`` vs ``experiment``).
    """
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(
            config_name="train_cached", overrides=[f"+cached_experiment={preset}", *overrides]
        )
    validate_cached_config(config)
    if OmegaConf.is_interpolation(config, "output_dir"):
        config.output_dir = "<automatic run directory>"
    return OmegaConf.to_yaml(config, resolve=True)


def _resolve_data_config(cfg: Mapping[str, Any]) -> RegionsDataConfig:
    fields = {f.name for f in dataclasses.fields(RegionsDataConfig)}
    kwargs = {key: value for key, value in dict(cfg["data"]).items() if key in fields}
    return RegionsDataConfig(**kwargs)


def _resolve_trainer_config(cfg: Mapping[str, Any]) -> TrainerConfig:
    """Merge ``cfg["trainer"]`` into a :class:`TrainerConfig`, the same way the main runner does.

    ``train_cached.yaml``'s ``trainer:`` section only ever sets fields that exist on
    :class:`TrainerConfig` (structured-config merge rejects anything else), so most of
    that dataclass's backbone/adapter-specific fields simply keep their defaults here --
    unused, since this model has no backbone to freeze or adapt.
    """
    try:
        trainer_cfg = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(TrainerConfig), cfg["trainer"])
        )
    except OmegaConfBaseException as exc:
        raise ValueError(f"Invalid trainer configuration: {exc}") from exc
    if not isinstance(trainer_cfg, TrainerConfig):
        raise TypeError("trainer configuration did not resolve to TrainerConfig")
    if trainer_cfg.resume_from_checkpoint and trainer_cfg.init_weights_from_checkpoint:
        raise ValueError(
            "Set only one of trainer.resume_from_checkpoint or trainer.init_weights_from_checkpoint"
        )
    return trainer_cfg


def _build_optimizer(model: RegionCountModel, trainer_cfg: TrainerConfig) -> AdamW:
    """AdamW with :class:`~regulonado.training.cached.model.CountHead` params excluded from
    weight decay.

    Shrinking a replicate offset or a noise scale toward zero is not regularisation of
    the sequence model -- see ``CountHead``'s docstring. 1-D parameters (norm/bias) are
    also excluded, matching the main runner's convention.
    """
    count_head_names = set(model.count_head_parameter_names())
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in count_head_names or parameter.ndim <= 1:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": trainer_cfg.learning_rate,
                "weight_decay": trainer_cfg.weight_decay,
                "name": "head",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": trainer_cfg.learning_rate,
                "weight_decay": 0.0,
                "name": "head_no_decay",
            }
        )
    if not groups:
        raise ValueError("model has no trainable parameters")
    return AdamW(groups)


def build_model(
    cfg: Mapping[str, Any],
    manifest: EmbeddingManifest,
    data: RegionCountData,
    group_names: list[str],
    track_groups: np.ndarray,
) -> RegionCountModel:
    """Build :class:`RegionCountModel` from the embeddings manifest, track table and config."""
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    data_cfg = cfg["data"]

    contrast_weight = float(loss_cfg.get("contrast_weight", 0.0) or 0.0)
    contrast_task_weights = None
    if contrast_weight:
        target_group = data_cfg.get("target_group")
        contrast_task_weight = loss_cfg.get("contrast_task_weight")
        if contrast_task_weight is not None:
            if not target_group:
                raise ValueError("loss.contrast_weight requires data.target_group")
            if target_group not in group_names:
                raise ValueError(f"data.target_group {target_group!r} not in groups {group_names}")
            contrast_task_weights = [
                float(contrast_task_weight) if name == target_group else 1.0
                for name in group_names
            ]

    config = RegionCountConfig(
        backbone_name=manifest.backbone,
        backbone_checkpoint=manifest.checkpoint,
        k=manifest.k,
        d=manifest.d,
        track_groups=[int(g) for g in track_groups],
        log_size_factors=[float(v) for v in data.log_size_factors()],
        track_names=list(data.track_names),
        group_names=list(group_names),
        hidden=int(model_cfg.get("hidden", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        eta_max=model_cfg.get("eta_max"),
        count_noise=str(model_cfg.get("count_noise", "nb")),
        count_replicate_offsets=bool(model_cfg.get("count_replicate_offsets", True)),
        count_freeze_noise=bool(model_cfg.get("count_freeze_noise", False)),
        count_noise_shrinkage=float(model_cfg.get("count_noise_shrinkage", 0.0)),
        loss_contrast_weight=contrast_weight,
        loss_contrast_multiplier=loss_cfg.get("contrast_multiplier"),
        contrast_task_weights=contrast_task_weights,
    )
    return RegionCountModel(config)


def _load_warm_start(model: RegionCountModel, checkpoint: str | Path) -> None:
    """Load model weights from *checkpoint*, skipping any tensor whose shape has changed.

    Unlike :func:`regulonado.training.runner.load_model_weights_only` (which raises on
    any missing/unexpected key), a later region-head stage may warm-start from an
    earlier one built against a different track/group set (a different embeddings run,
    or ``data.specific_only``/track-table change) -- only :class:`CountHead`'s
    ``track_groups``-shaped buffers and parameters would then disagree in shape. Loading
    non-strictly and skipping shape mismatches keeps the rest of the head (pooling,
    MLP) warm-started while letting a resized ``CountHead`` re-initialise.
    """
    checkpoint_path = Path(checkpoint)
    weight_path = checkpoint_path
    if checkpoint_path.is_dir():
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

    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for name, tensor in state_dict.items():
        target = model_state.get(name)
        if target is not None and target.shape == tensor.shape:
            compatible[name] = tensor
        else:
            skipped.append(name)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if skipped:
        logger.warning(
            f"warm start: skipped {len(skipped)} shape-mismatched tensor(s): {skipped[:10]}"
        )
    unexpected = [name for name in unexpected if not name.endswith("num_batches_tracked")]
    if unexpected:
        logger.warning(f"warm start: unexpected checkpoint tensor(s): {unexpected[:10]}")
    if missing:
        logger.warning(
            f"warm start: checkpoint has no value for {len(missing)} tensor(s): {missing[:10]}"
        )


def _load_and_prepare(
    cfg: Mapping[str, Any],
) -> tuple[PreparedRegionData, EmbeddingManifest, RegionsDataConfig]:
    data_cfg = _resolve_data_config(cfg)
    if not data_cfg.path:
        raise ValueError("data.path is required")
    if not data_cfg.embeddings_dir:
        raise ValueError("data.embeddings_dir is required")
    raw = RegionCountData.read(Path(data_cfg.path))
    data, manifest = attach_region_rows(raw, Path(data_cfg.embeddings_dir))
    prepared = prepare_region_data(data, data_cfg)
    return prepared, manifest, data_cfg


def _build_datasets(
    prepared: PreparedRegionData,
    data_cfg: RegionsDataConfig,
    trainer_cfg: TrainerConfig,
    *,
    seed: int,
) -> tuple[CachedRegionDataset, Any, bool]:
    split_names = set(prepared.data.regions["split"].unique().to_list())
    has_eval = "val" in split_names

    store = EmbeddingStore(data_cfg.embeddings_dir)
    train_dataset = CachedRegionDataset(
        prepared.data,
        store,
        split="train",
        train=True,
        sample_weights=prepared.train_sample_weights,
        enable_rc_aug=data_cfg.enable_rc_aug,
        preload=data_cfg.preload,
        drop_missing_from_cache=data_cfg.drop_missing_from_cache,
    )
    val_dataset: Any = None
    if has_eval:
        val_dataset = CachedRegionDataset(
            prepared.data,
            store,
            split="val",
            train=False,
            enable_rc_aug=False,
            preload=data_cfg.preload,
            drop_missing_from_cache=data_cfg.drop_missing_from_cache,
        )
        if trainer_cfg.max_eval_samples is not None:
            n_val = len(val_dataset)
            if n_val > trainer_cfg.max_eval_samples:
                rng = np.random.default_rng(seed)
                indices = sorted(
                    rng.choice(n_val, size=trainer_cfg.max_eval_samples, replace=False).tolist()
                )
                val_dataset = Subset(val_dataset, indices)
    return train_dataset, val_dataset, has_eval


def run_training(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Train one :class:`RegionCountModel` end to end from a resolved Hydra config.

    Parameters
    ----------
    cfg
        Complete region-training configuration: ``data``/``model``/``loss``/``trainer``
        sections plus ``output_dir`` and ``seed`` (see ``python/configs/train_cached.yaml``).

    Returns
    -------
    A summary dict (also written to ``<output_dir>/training_summary.json``): output
    directory, seed, track/group counts, and the train/eval loss history.
    """
    validate_cached_config(cfg)
    trainer_cfg = _resolve_trainer_config(cfg)
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)

    prepared, manifest, data_cfg = _load_and_prepare(cfg)
    train_dataset, val_dataset, has_eval = _build_datasets(
        prepared, data_cfg, trainer_cfg, seed=seed
    )

    schedule: TrainingSchedule = _resolve_training_schedule(
        trainer_cfg, train_rows=len(train_dataset), world_size=_world_size(), has_eval=has_eval
    )
    logger.info(
        f"schedule: steps_per_epoch={schedule.steps_per_epoch} total_steps={schedule.total_steps} "
        f"eval_steps={schedule.eval_steps} checkpoint_steps={schedule.checkpoint_steps}"
    )

    model = build_model(cfg, manifest, prepared.data, prepared.group_names, prepared.track_groups)
    if trainer_cfg.init_weights_from_checkpoint:
        _load_warm_start(model, trainer_cfg.init_weights_from_checkpoint)

    output_dir = Path(str(cfg.get("output_dir") or Path.cwd() / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict: dict[str, Any] = {"train": train_dataset}
    if val_dataset is not None:
        dataset_dict["val"] = val_dataset
    write_provenance(
        output_dir=output_dir,
        cfg=cfg,
        data_path=Path(data_cfg.path),
        dataset_dict=dataset_dict,
        metadata={"backbone": manifest.backbone, "k": manifest.k, "d": manifest.d},
        records=prepared.data.tracks.to_dicts(),
        trainer_cfg=trainer_cfg,
    )

    optimizer = _build_optimizer(model, trainer_cfg)
    scheduler = _build_scheduler_for_trainer(optimizer, trainer_cfg, schedule)
    training_args = _build_training_arguments(
        output_dir, trainer_cfg, schedule, has_eval=has_eval, seed=seed
    )

    compute_metrics = None
    if has_eval:
        compute_metrics = GroupedCountMetrics(
            group_names=prepared.group_names,
            track_groups=prepared.track_groups,
            log_size_factors=prepared.data.log_size_factors(),
            top_decile_task=data_cfg.target_group,
        )

    callbacks: list = [LRLogCallback()]
    if "wandb" in trainer_cfg.report_to:
        callbacks.append(WandbConfigCallback(cfg))
    if trainer_cfg.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=trainer_cfg.early_stopping_patience,
                early_stopping_threshold=trainer_cfg.early_stopping_threshold,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    resume = trainer_cfg.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume if resume else None)
    trainer.save_model(str(output_dir))
    trainer.save_state()

    train_losses = [
        float(entry["loss"])
        for entry in trainer.state.log_history
        if "loss" in entry and "eval_loss" not in entry
    ]
    eval_losses = [
        float(entry["eval_loss"]) for entry in trainer.state.log_history if "eval_loss" in entry
    ]
    history = {"train/loss": train_losses, "eval/loss": eval_losses}

    summary = {
        "output_dir": str(output_dir),
        "seed": seed,
        "backbone": manifest.backbone,
        "n_tracks": prepared.data.n_tracks,
        "n_groups": len(prepared.group_names),
        "group_names": prepared.group_names,
        "resume_from_checkpoint": trainer_cfg.resume_from_checkpoint,
        "init_weights_from_checkpoint": trainer_cfg.init_weights_from_checkpoint,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_cached")
def hydra_entrypoint(cfg: DictConfig) -> None:
    """Hydra entrypoint: resolve ``python/configs/train_cached.yaml``, then :func:`run_training`."""
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Hydra config did not resolve to a dictionary")
    run_training(config)


def main() -> None:
    hydra_entrypoint()


if __name__ == "__main__":
    main()
