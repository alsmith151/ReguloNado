"""Discovery and inspection helpers for Hydra experiment configs.

The Typer CLI (`regulonado experiments`, `regulonado train`) uses these helpers
to list, describe, and validate experiment names without forcing the user to
know them in advance. Experiment YAMLs live in two places, mirroring the
`hydra.searchpath` used by ``scripts/train_slurm.sh``:

* the built-in ``python/configs/experiment/`` directory shipped with the package, and
* an optional run-specific ``scripts/experiment/`` directory at the repo root.

Each experiment YAML carries a human description in its leading ``#`` comment
block; the first comment line (after the ``# @package _global_`` marker) is
treated as a one-line summary.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ExperimentInfo:
    """Metadata for a single experiment config."""

    name: str
    path: Path
    summary: str
    description: str
    builtin: bool


def _package_root() -> Path:
    """Directory of the installed ``regulonado`` package (``python/regulonado``)."""
    return Path(__file__).resolve().parent


def builtin_experiment_dir() -> Path:
    """Built-in ``python/configs/experiment`` directory."""
    return _package_root().parent / "configs" / "experiment"


def repo_root() -> Optional[Path]:
    """Repo root inferred from the package location, or ``None`` if not a checkout.

    The package lives at ``<repo>/python/regulonado``; the repo root is two
    levels up. Returns ``None`` when that layout does not hold (e.g. an
    installed wheel without the surrounding source tree).
    """
    candidate = _package_root().parents[1]
    return candidate if (candidate / "scripts").is_dir() else None


def experiment_dirs() -> list[Path]:
    """Existing experiment search dirs, built-ins first then repo overrides."""
    dirs: list[Path] = []
    builtin = builtin_experiment_dir()
    if builtin.is_dir():
        dirs.append(builtin)
    root = repo_root()
    if root is not None:
        override = root / "scripts" / "experiment"
        if override.is_dir():
            dirs.append(override)
    return dirs


def _parse_header(path: Path) -> tuple[str, str]:
    """Return ``(summary, description)`` parsed from a YAML's leading comments.

    Reads the contiguous block of ``#`` comment lines at the top of the file,
    drops the ``# @package ...`` directive, and uses the first remaining comment
    line as the summary. The description is the whole comment block rejoined.
    """
    comment_lines: list[str] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                stripped = raw.strip()
                if stripped.startswith("#"):
                    text = stripped[1:].strip()
                    if text.startswith("@package"):
                        continue
                    comment_lines.append(text)
                elif stripped == "":
                    # Allow blank lines within the leading comment block.
                    if comment_lines:
                        comment_lines.append("")
                    continue
                else:
                    break
    except OSError:
        return ("", "")

    # Trim trailing blank lines.
    while comment_lines and comment_lines[-1] == "":
        comment_lines.pop()

    summary = next((line for line in comment_lines if line), "")
    description = "\n".join(comment_lines).strip()
    return (summary, description)


def discover_experiments() -> dict[str, ExperimentInfo]:
    """Map experiment name -> info, built-ins taking precedence over overrides."""
    found: dict[str, ExperimentInfo] = {}
    builtin = builtin_experiment_dir()
    for directory in experiment_dirs():
        is_builtin = directory == builtin
        for path in sorted(directory.glob("*.yaml")):
            name = path.stem
            if name in found:
                # Built-ins are listed first, so keep the earlier entry.
                continue
            summary, description = _parse_header(path)
            found[name] = ExperimentInfo(
                name=name,
                path=path,
                summary=summary,
                description=description,
                builtin=is_builtin,
            )
    return found


def suggest_experiments(name: str, limit: int = 3) -> list[str]:
    """Close-match experiment names for a (likely mistyped) ``name``."""
    names = list(discover_experiments())
    return difflib.get_close_matches(name, names, n=limit)


def resolve_experiment(name: str) -> dict:
    """Compose the effective config for ``name`` via the Hydra Compose API.

    Returns the resolved config as a plain dict so callers can read the
    composed ``backbone``/``head``/``loss`` selections and ``trainer`` values.
    Hydra is imported lazily so plain discovery (listing) has no heavy deps.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = str((_package_root().parent / "configs").resolve())
    overrides = [f"+experiment={name}"]
    extra = repo_root()
    if extra is not None and (extra / "scripts" / "experiment").is_dir():
        # Mirror scripts/train_slurm.sh so run-specific experiments resolve too.
        overrides.insert(0, f"+hydra.searchpath=[file://{extra / 'scripts'}]")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="train", overrides=overrides)
    resolved = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(resolved, dict)
    return resolved
