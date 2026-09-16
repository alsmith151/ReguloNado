"""Canonical workflow/sweep settings merging and Hydra override rendering."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def flatten_settings(settings: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten canonical nested settings, rejecting ambiguous dotted input keys."""
    flattened: dict[str, Any] = {}
    for key, value in settings.items():
        if "." in key:
            raise ValueError(
                f"Training setting {key!r} uses obsolete dotted YAML syntax; "
                "use nested mappings instead"
            )
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(flatten_settings(value, name))
        else:
            flattened[name] = value
    return flattened


def merge_training_settings(
    layers: Sequence[Mapping[str, Any]],
    *,
    seed: int | None = None,
    pretrained_model: str | None = None,
) -> dict[str, Any]:
    """Merge common/phase/run settings using the workflow's declared precedence."""
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(flatten_settings(layer))
    if seed is not None:
        merged["seed"] = seed
    if pretrained_model is not None:
        merged["backbone.pretrained_name"] = pretrained_model
    return merged


def hydra_override_items(settings: Mapping[str, Any]) -> list[str]:
    """Render deterministic Hydra overrides with config groups before their fields.

    Top-level keys select config groups or ordinary root values. Dotted keys use
    ``++`` so fields specific to the selected group can be set without weakening
    structured-config validation of the final composed configuration.
    """
    group_or_root = sorted((key, value) for key, value in settings.items() if "." not in key)
    dotted = sorted((key, value) for key, value in settings.items() if "." in key)
    return [
        f"{key if '.' not in key else f'++{key}'}={json.dumps(value, separators=(',', ':'))}"
        for key, value in [*group_or_root, *dotted]
    ]
