"""Thin wrapper over ``peft`` for LoRA (attention) + LoCon (convolution) PEFT.

``peft``'s LoRA decomposition applied to ``nn.Conv1d`` *is* the LoCon method
from the paper (two consecutive convolutions, kernel size ``k`` then ``1``) —
see ``peft/tuners/lora/layer.py``'s ``Conv1d`` LoRA branch. So a single
``peft.LoraConfig`` whose ``target_modules`` names both attention ``Linear``
modules and trunk ``Conv1d`` modules gives LoRA and LoCon together; no
separate lycoris/loralib dependency is needed.

Uses ``peft.LoraModel`` directly, not ``get_peft_model``/
``inject_adapter_in_model``: it injects adapters in place (submodules are
swapped via ``setattr``, so ``backbone.model`` keeps its identity and
``forward_features``'s call to ``self.model.get_embs_after_crop(...)`` keeps
working), and it is the only one of the three that exposes
``merge_and_unload()``.

Config types are accepted structurally (``Any``) rather than imported, so
this module stays decoupled from ``regulonado.training.config.AdapterConfig``
(owned separately): any object exposing the expected attributes works.
"""

from __future__ import annotations

from typing import Any

from peft import LoraConfig, LoraModel


def _target_exists(target: str, module_names: set[str]) -> bool:
    """True if ``target`` names an exact module or the tail of a dotted module path.

    Mirrors how ``peft`` itself resolves ``target_modules`` entries (see
    ``peft.utils.other.get_pattern_key``): a bare leaf name like ``"to_q"``
    matches every module whose full dotted path ends with ``".to_q"`` (or
    equals ``"to_q"``), while a full dotted path like
    ``"res_tower.6.conv_layer"`` matches only itself.
    """
    return any(name == target or name.endswith(f".{target}") for name in module_names)


def resolve_lora_targets(backbone_adapter: Any, adapter_cfg: Any) -> list[str]:
    """Resolve peft ``target_modules`` names for LoRA (attention) + LoCon (conv).

    Walks the live ``backbone_adapter.model`` tree; does not rely on any
    hardcoded assumption about which attention flavour is present.

    Attention
    ---------
    flashzoi fuses q/k/v into a single ``mha.Wqkv`` projection (GQA) plus
    ``mha.out_proj``; plain (non-flash) Borzoi keeps separate ``to_q``/
    ``to_k``/``to_v``/``to_out`` projections. Both are detected by
    introspection and supported:

    - flashzoi: always targets ``Wqkv`` + ``out_proj``. Because q/k/v are
      fused into one projection, the paper's "default mode = adapt q and v
      only" is not expressible here — adapting ``Wqkv`` necessarily adapts
      all three. The paper reports performance is robust to default vs. full
      mode, so this is accepted rather than worked around.
    - non-flash: ``attention_mode="default"`` targets ``to_q``, ``to_v``
      only; ``"full"`` also adds ``to_k``, ``to_out``.

    Convolution (LoCon)
    --------------------
    The last ``adapter_cfg.locon_conv_blocks`` entries of
    ``backbone_adapter.iter_locon_conv_candidates()`` (data-flow order), with
    ``.conv_layer`` appended to reach the actual ``nn.Conv1d``.
    ``locon_conv_blocks=0`` disables LoCon entirely.

    When ``locon_conv_blocks`` selects *every* candidate, this replicates
    Baskerville's ``conv1_tune``: the first candidate (``conv_dna``) is
    excluded from LoCon here (gated on
    ``adapter_cfg.tune_first_conv_when_all``) so the caller
    (:func:`attach_adapters`) can unfreeze it fully instead of LoRA-adapting
    it.

    Raises
    ------
    ValueError
        If the resolved target list is empty, or if any resolved target does
        not exist on ``backbone_adapter.model``'s module tree. A silent
        no-op (peft matching nothing and training proceeding as if frozen)
        is the failure mode this guards against.
    """
    model = backbone_adapter.model
    module_names = {name for name, _ in model.named_modules() if name}

    targets: list[str] = []

    if adapter_cfg.attention:
        has_fused_qkv = any(name.endswith("Wqkv") for name in module_names)
        has_separate_qkv = any(name.endswith("to_q") for name in module_names)
        if has_fused_qkv:
            targets += ["Wqkv", "out_proj"]
        elif has_separate_qkv:
            targets += ["to_q", "to_v"]
            if adapter_cfg.attention_mode == "full":
                targets += ["to_k", "to_out"]
        else:
            raise ValueError(
                "adapter_cfg.attention=True but neither a fused 'Wqkv' (flashzoi) nor "
                "separate 'to_q' (plain Borzoi) attention projection was found on the "
                "backbone module tree."
            )

    candidates = list(backbone_adapter.iter_locon_conv_candidates())
    n = adapter_cfg.locon_conv_blocks
    if n > 0:
        if n > len(candidates):
            raise ValueError(
                f"adapter_cfg.locon_conv_blocks={n} exceeds the number of available "
                f"LoCon candidates ({len(candidates)})."
            )
        selected = candidates[-n:]
        if n == len(candidates) and adapter_cfg.tune_first_conv_when_all and candidates:
            # Baskerville's conv1_tune: when every candidate is selected, exclude the
            # first (conv_dna) from LoCon; attach_adapters unfreezes it fully instead.
            first_candidate = candidates[0]
            selected = [name for name in selected if name != first_candidate]
        targets += [f"{name}.conv_layer" for name in selected]

    if not targets:
        raise ValueError(
            "resolve_lora_targets resolved an empty target list — refusing a silent "
            "no-op. Check adapter_cfg.attention/locon_conv_blocks."
        )

    for target in targets:
        if not _target_exists(target, module_names):
            raise ValueError(
                f"Resolved LoRA/LoCon target {target!r} does not exist on the backbone "
                "module tree; refusing a silent no-op."
            )

    return targets


def attach_adapters(model: Any, adapter_cfg: Any) -> LoraModel:
    """Inject LoRA (attention) + LoCon (conv) adapters into ``model.backbone.model``.

    Builds a single ``peft.LoraConfig`` and returns
    ``LoraModel(model.backbone.model, cfg, "default")``. Attention targets use
    the base ``r``/``lora_alpha``; conv (LoCon) targets get their own
    rank/alpha via peft's ``rank_pattern``/``alpha_pattern`` fields, matched
    by the ``"conv_layer"`` suffix all conv targets share (see
    :func:`resolve_lora_targets`).

    Raises
    ------
    ValueError
        If ``adapter_cfg.locon_conv_blocks > 0`` while ``adapter_cfg.attention``
        is ``False``. Baskerville's own ``add_locon`` always calls
        ``add_lora`` first — LoCon is never used without attention LoRA — so
        this is enforced rather than silently allowed.
    """
    if not adapter_cfg.attention and adapter_cfg.locon_conv_blocks > 0:
        raise ValueError(
            "adapter_cfg.locon_conv_blocks > 0 requires adapter_cfg.attention=True "
            "(LoCon is never attached without attention LoRA — see Baskerville's "
            "add_locon, which always calls add_lora first)."
        )

    backbone_adapter = model.backbone
    targets = resolve_lora_targets(backbone_adapter, adapter_cfg)

    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    if adapter_cfg.locon_conv_blocks > 0:
        rank_pattern["conv_layer"] = adapter_cfg.locon_r
        alpha_pattern["conv_layer"] = adapter_cfg.locon_alpha

    lora_config = LoraConfig(
        r=adapter_cfg.lora_r,
        lora_alpha=adapter_cfg.lora_alpha,
        lora_dropout=adapter_cfg.dropout,
        target_modules=targets,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        bias="none",
    )
    lora_model = LoraModel(backbone_adapter.model, lora_config, "default")

    candidates = list(backbone_adapter.iter_locon_conv_candidates())
    n = adapter_cfg.locon_conv_blocks
    if n > 0 and n == len(candidates) and adapter_cfg.tune_first_conv_when_all and candidates:
        first_conv = dict(backbone_adapter.model.named_modules())[candidates[0]]
        for parameter in first_conv.parameters():
            parameter.requires_grad = True

    return lora_model


def merge_adapters(lora_model: LoraModel) -> None:
    """Fold adapters into base weights in place, so inference needs zero changes."""
    lora_model.merge_and_unload()
