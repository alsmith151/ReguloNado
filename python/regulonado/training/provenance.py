from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from regulonado.training.config import TrainerConfig


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _run_git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_summary(dataset_dict: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, split in dataset_dict.items():
        try:
            summary[split_name] = {"num_rows": len(split)}
        except TypeError:
            summary[split_name] = {"num_rows": None, "streaming": True}
    return summary


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_summary() -> dict[str, Any]:
    slurm_keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_SUBMIT_DIR",
        "SLURM_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
        "CUDA_VISIBLE_DEVICES",
    ]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        },
        "packages": {
            name: _package_version(name)
            for name in ("transformers", "datasets", "accelerate", "wandb", "numpy", "torch")
        },
        "slurm": {key: os.environ[key] for key in slurm_keys if key in os.environ},
    }


def _write_provenance(
    *,
    output_dir: Path,
    cfg: Mapping[str, Any],
    data_path: Path,
    dataset_dict: Mapping[str, Any],
    metadata: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    trainer_cfg: TrainerConfig,
) -> None:
    if not trainer_cfg.provenance.enabled:
        return

    repo = Path.cwd()
    metadata_candidates = [
        data_path / "regulonado_metadata.json",
        data_path / "track_metadata.json",
    ]
    metadata_path = next((path for path in metadata_candidates if path.exists()), None)
    git_status = _run_git(repo, "status", "--short")
    provenance = {
        "config": cfg,
        "command": " ".join(sys.argv),
        "git": {
            "branch": _run_git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _run_git(repo, "rev-parse", "HEAD"),
            "dirty": bool(git_status),
            "status_short": git_status,
        },
        "dataset": {
            "path": str(data_path),
            "metadata_path": str(metadata_path) if metadata_path is not None else None,
            "metadata_sha256": _file_sha256(metadata_path) if metadata_path is not None else None,
            "n_tracks": len(records),
            "splits": _split_summary(dataset_dict),
            "context_length": metadata.get("context_length"),
            "n_pred_bins": metadata.get("n_pred_bins"),
            "bin_size": metadata.get("bin_size"),
        },
        "checkpoint_reuse": {
            "resume_from_checkpoint": trainer_cfg.resume_from_checkpoint,
            "init_weights_from_checkpoint": trainer_cfg.init_weights_from_checkpoint,
        },
        "environment": _environment_summary(),
    }
    _write_json(output_dir / "provenance.json", provenance)
    _write_json(output_dir / "resolved_config.json", cfg)
    if trainer_cfg.provenance.save_git_diff:
        diff = _run_git(repo, "diff", "--no-ext-diff")
        if diff:
            (output_dir / "git_diff.patch").write_text(diff)
