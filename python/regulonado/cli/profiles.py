"""Snakemake execution preset discovery, shared with SeqNado.

SeqNado installs its presets into ``~/.config/snakemake/profile_<words>/`` and
addresses them by a shortcode built from the initials of the words after
``profile_`` (``profile_local_environment`` -> ``le``). ReguloNado uses the same
directory and the same rule, so a user running both pipelines on one machine
keeps a single set of cluster/local execution configs.

When SeqNado is installed we call its resolver directly rather than duplicating
it. When it is not, the fallback below reimplements the same lookup — the scheme
is a filesystem convention, so behaviour is identical either way.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

# Presets shipped by this package, under workflow/profiles/.
_PACKAGED_PROFILES = "regulonado.workflow.profiles"

# SeqNado presets whose default resources are far too small for ReguloNado's
# dataset-building and training rules. Selecting one still works, but silently
# running a multi-hour GPU job under `mem: 3G, runtime: 1h` is not a useful
# default, so we say so once.
_UNDERSIZED_PRESETS = {"ss", "le", "lc", "ls", "ld"}

_GPU_RULES = ("build_dataset", "train_phase")


def profile_shortcode(name: str) -> str | None:
    """Return the shortcode for a ``profile_*`` directory name.

    Mirrors ``seqnado.utils.get_profile_name``: initials of the underscore-
    separated words following the ``profile_`` prefix.
    """
    if not name.startswith("profile_"):
        return None
    parts = name.split("_")[1:]
    initials = "".join(part[0] for part in parts if part)
    return initials or None


def _packaged_profiles() -> dict[str, str]:
    """Shortcode -> directory name for presets bundled with ReguloNado."""
    try:
        traversable = resources.files(_PACKAGED_PROFILES)
    except (ModuleNotFoundError, TypeError):  # pragma: no cover - packaging guard
        return {}

    profiles: dict[str, str] = {}
    for entry in traversable.iterdir():
        if not entry.is_dir():
            continue
        code = profile_shortcode(entry.name)
        if code:
            profiles[code] = entry.name
    return profiles


def _seqnado_profiles() -> dict[str, str]:
    """Shortcode -> directory name for presets provided by SeqNado, if installed."""
    try:
        from seqnado.utils import get_preset_profiles
    except ImportError:
        return {}
    try:
        return {code: name for code, name in get_preset_profiles().items() if code}
    except Exception as exc:  # SeqNado present but its resources are unreadable.
        logger.debug(f"Could not read SeqNado presets: {exc}")
        return {}


def get_preset_profiles() -> dict[str, str]:
    """Return every known preset as ``shortcode -> profile directory name``.

    SeqNado's presets are included when it is installed. On a shortcode clash
    ReguloNado's own preset wins, because its resource requests are the ones
    sized for this workflow.
    """
    profiles = dict(_seqnado_profiles())
    for code, name in _packaged_profiles().items():
        existing = profiles.get(code)
        if existing and existing != name:
            logger.debug(f"Preset '{code}' resolves to ReguloNado's {name} (not {existing})")
        profiles[code] = name
    return profiles


def _user_profile_dir() -> Path:
    return Path.home() / ".config" / "snakemake"


def resolve_profile_path(preset: str | None) -> Path | None:
    """Resolve a preset shortcode to a profile directory.

    Prefers the user's installed copy in ``~/.config/snakemake/`` — the one
    ``regulonado init`` (or ``seqnado init``) writes and the user edits — over the
    read-only packaged copy. Returns ``None`` when the shortcode is unknown.
    """
    if not preset:
        return None

    profiles = get_preset_profiles()
    directory = profiles.get(preset.lower())
    if not directory:
        return None

    user_path = _user_profile_dir() / directory
    if user_path.exists():
        return user_path

    packaged = _packaged_profiles()
    if directory in packaged.values():
        traversable = resources.files(_PACKAGED_PROFILES).joinpath(directory)
        with resources.as_file(traversable) as path:
            # Packaged profiles are plain directories on disk for a normal
            # install; materialising them is only needed for zipped installs,
            # where the caller must read the config before the context exits.
            return Path(path)

    # Known to SeqNado but not installed to ~/.config/snakemake and not ours.
    # SeqNado's resolver only falls back to its bundled copy when handed its own
    # package root, so pass it — otherwise every preset we advertise from SeqNado
    # but the user has not run `seqnado init` for would resolve to nothing.
    try:
        from seqnado.utils import resolve_profile_path as _seqnado_resolve

        resolved = _seqnado_resolve(preset, resources.files("seqnado"))
    except ImportError:
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"SeqNado could not resolve preset '{preset}': {exc}")
        return None

    if resolved is None:
        return None
    if isinstance(resolved, Path):
        return resolved
    with resources.as_file(resolved) as path:
        return Path(path)


def warn_if_undersized(preset: str, values: dict) -> None:
    """Warn when a shared preset has no resources sized for the heavy rules.

    SeqNado's presets default to ``mem: 3G, runtime: 1h`` — correct for its own
    alignment jobs, hopeless for dataset building or model training. We do not
    override the user's choice, but running a 24-hour GPU job under those
    defaults should never happen silently.
    """
    if preset.lower() not in _UNDERSIZED_PRESETS:
        return
    set_resources = values.get("set-resources") or {}
    if any(rule in set_resources for rule in _GPU_RULES):
        return
    logger.warning(
        f"Preset '{preset}' has no set-resources for {' or '.join(_GPU_RULES)}; "
        "its defaults are sized for SeqNado's alignment jobs, not for dataset "
        "building or training. Use '--preset sg' or add set-resources for those rules."
    )


def format_available_presets() -> str:
    """Human-readable preset list for CLI help and error messages."""
    profiles = get_preset_profiles()
    if not profiles:
        return "none found"
    return ", ".join(f"{code} ({name})" for code, name in sorted(profiles.items()))
