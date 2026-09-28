"""Load a trained :class:`~regulonado.training.regions.model.RegionCountModel` for inference.

Two pieces:

- :func:`load_state_dict_non_strict`, the shape-tolerant checkpoint loader shared with
  :mod:`regulonado.training.regions.runner`'s warm-start path -- warn-only, since a
  later region-head stage may legitimately warm-start from a checkpoint with a
  different track/group set.
- :func:`load_region_count_model`, which rebuilds a *live-trunk* model end to end
  (backbone, fine-tune/adapter wiring, head) from a saved checkpoint directory and
  loads its weights **strictly** -- attribution and design tooling need every trained
  parameter actually restored, so a shape mismatch or missing key there is a bug, not
  a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from regulonado.training.regions.model import RegionCountConfig, RegionCountModel

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

__all__ = ["load_region_count_model", "load_state_dict_non_strict"]


def load_state_dict_non_strict(
    model: "nn.Module",
    checkpoint: str | Path,
    *,
    allow_missing: tuple[str, ...] = (),
    strict_unexpected: bool = True,
) -> tuple[list[str], list[str]]:
    """Load *checkpoint*'s weights into *model*, skipping any tensor whose shape has changed.

    *checkpoint* may be a checkpoint directory (``model.safetensors`` or
    ``pytorch_model.bin`` inside it) or a weights file directly. Shape-mismatched
    entries are dropped before ``load_state_dict(strict=False)`` runs, so the rest of
    the model still loads even when, say, a resized head disagrees in shape with the
    checkpoint.

    This function itself never raises on missing/unexpected keys -- it only loads and
    reports them, via the returned ``(missing, unexpected)`` and a warning log for each
    non-empty case. *allow_missing* and *strict_unexpected* are unused by this
    warn-only path; they exist so a caller that wants stricter behaviour (e.g.
    :func:`load_region_count_model`) can filter/assert against the same two lists this
    function already computes, without a second implementation of the loading logic.

    Returns:
        ``(missing, unexpected)``: checkpoint-relative names of tensors *model* has
        that the checkpoint didn't supply (or skipped for a shape mismatch), and
        tensors the checkpoint had that *model* doesn't (``*.num_batches_tracked``
        buffers are dropped from ``unexpected``, since those are BatchNorm's own
        bookkeeping and not meaningfully "unexpected").
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
    shape_mismatched: list[str] = []
    # A tensor the model has no entry for at all is genuinely "unexpected" (the
    # checkpoint doesn't belong to this architecture); one the model has under the same
    # name but a different shape is "shape-mismatched" (e.g. a resized CountHead) --
    # distinct cases, so they're tracked separately rather than both folded into
    # `load_state_dict`'s own `unexpected`, which never fires here: `compatible`'s keys
    # are by construction a subset of the model's, so passing it to `load_state_dict`
    # can only ever report `missing`, never `unexpected`.
    unexpected: list[str] = []
    for name, tensor in state_dict.items():
        target = model_state.get(name)
        if target is None:
            unexpected.append(name)
        elif target.shape == tensor.shape:
            compatible[name] = tensor
        else:
            shape_mismatched.append(name)
    missing, _ = model.load_state_dict(compatible, strict=False)
    if shape_mismatched:
        logger.warning(
            f"warm start: skipped {len(shape_mismatched)} shape-mismatched tensor(s): "
            f"{shape_mismatched[:10]}"
        )
    unexpected = [name for name in unexpected if not name.endswith("num_batches_tracked")]
    if unexpected:
        logger.warning(f"warm start: unexpected checkpoint tensor(s): {unexpected[:10]}")
    if missing:
        logger.warning(
            f"warm start: checkpoint has no value for {len(missing)} tensor(s): {missing[:10]}"
        )
    return missing, unexpected


def load_region_count_model(
    checkpoint_dir: str | Path, device: str | torch.device | None = None
) -> RegionCountModel:
    """Rebuild a live-trunk :class:`RegionCountModel` from a saved checkpoint directory.

    Unlike a plain ``RegionCountModel.from_pretrained`` (which only knows how to build
    the head), this reconstructs the trunk too: the backbone adapter named in
    ``config.trunk["backbone"]``, wired up by :func:`~regulonado.training.regions.live.
    prepare_trunk` exactly as training left it (frozen, fully fine-tuned, or with
    LoCon/other adapter submodules) -- those adapter submodules must exist *before*
    weights are loaded, or their parameter names won't be in the model to load into.
    ``gradient_checkpointing`` is forced off: it is pure overhead at inference and
    interacts badly with ``torch.inference_mode``.

    Loading is **strict** here, unlike :func:`load_state_dict_non_strict`'s own
    warn-only default: every checkpoint tensor must land somewhere in the model
    (``unexpected`` empty) and every model tensor the checkpoint didn't supply must be
    one of the frozen trunk buffers this model deliberately never saves
    (``model._keys_to_ignore_on_save``) or a ``*.num_batches_tracked`` buffer. A silent
    partial load here would score the bare pretrained backbone with the head's
    predictions layered on top of un-fine-tuned features, with no symptom at all.

    Raises:
        ValueError: if *checkpoint_dir* has no ``trunk`` config (a cached-embeddings
            head, which has no backbone to rebuild and so cannot run this path), or if
            loading leaves unexpected or genuinely-missing tensors.
    """
    from regulonado.model.adapters import BackboneSpec, build_backbone_adapter
    from regulonado.training.regions.live import (
        LiveTrunk,
        TrunkFinetuneConfig,
        TrunkWindow,
        prepare_trunk,
    )

    checkpoint_dir = Path(checkpoint_dir)
    config = RegionCountConfig.from_pretrained(checkpoint_dir)
    if config.trunk is None:
        raise ValueError(
            f"{checkpoint_dir} is a cached-embeddings region head (no `trunk:` block in "
            "config.json): it has no backbone to rebuild. Attribution and design need a "
            "live trunk to run the backbone on arbitrary sequence -- retrain this head "
            "with data.fasta set (a live-trunk run) instead of data.embeddings_dir."
        )

    backbone_cfg = config.trunk["backbone"]
    spec = BackboneSpec(
        backbone_type=backbone_cfg["type"],
        pretrained_name=backbone_cfg.get("pretrained"),
        features=backbone_cfg.get("features", "trunk"),
        allow_random_init=backbone_cfg.get("pretrained") is None,
    )
    adapter = build_backbone_adapter(spec)

    finetune = TrunkFinetuneConfig(**{**config.trunk["finetune"], "gradient_checkpointing": False})
    prepare_trunk(adapter, finetune)

    trunk = LiveTrunk(adapter, TrunkWindow(**config.trunk["window"]))
    model = RegionCountModel(config, trunk=trunk)

    missing, unexpected = load_state_dict_non_strict(model, checkpoint_dir)
    if unexpected:
        raise ValueError(
            f"{checkpoint_dir}: checkpoint has tensor(s) the rebuilt model does not: "
            f"{unexpected}"
        )
    ignorable = set(model._keys_to_ignore_on_save or [])
    bad_missing = [
        name
        for name in missing
        if name not in ignorable and not name.endswith("num_batches_tracked")
    ]
    if bad_missing:
        raise ValueError(
            f"{checkpoint_dir}: checkpoint is missing trained tensor(s) the rebuilt model "
            f"needs: {bad_missing}"
        )

    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    restored = sum(p.numel() for name, p in model.named_parameters() if name not in missing)
    logger.info(
        f"loaded region count model from {checkpoint_dir}: n_groups={config.n_groups} "
        f"group_names={config.group_names} window={config.trunk['window']} "
        f"loss_contrast_multiplier={config.loss_contrast_multiplier} "
        f"finetune={finetune.finetune} restored_parameters={restored:,}"
    )
    return model
