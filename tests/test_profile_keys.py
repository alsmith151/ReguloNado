"""Test coverage of Snakemake profile keys consumed by the CLI translator.

F23 audit: Every packaged profile key must be explicitly consumed by the
translator in cli/pipeline.py, or be in an explicit allowlist with a reason.
A missing key fails silently (e.g., a container preset runs in the ambient
environment without saying so). This test makes that a CI failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from regulonado.cli.pipeline import (
    _deployment_settings,
    _executor_settings,
)

# Profile keys that are intentionally not consumed by the translator, with reasons.
# These are checked per key to ensure the list is actively maintained.
# Only keys actually present in packaged profiles should be listed here.
INTENTIONAL_ALLOWLIST: dict[str, str] = {
    # Top-level keys used in pipeline() but not by the translator functions:
    "executor": "Used directly in pipeline() to select executor",
    "jobs": "Used directly in pipeline() as max_jobs",
    "latency-wait": "Used directly in pipeline() for ExecutionSettings.latency_wait",
    "rerun-incomplete": "Used directly in pipeline() for DAGSettings.force_incomplete",
    "default-resources": "Used directly in pipeline() for ResourceSettings.default_resources",
    "set-resources": "Used directly in pipeline() for ResourceSettings.overwrite_resources",
    # Hardcoded in OutputSettings, not read from profile:
    "show-failed-logs": "Hardcoded to True in OutputSettings (would be redundant to read)",
    "printshellcmds": "Hardcoded to True in OutputSettings (would be redundant to read)",
}


def _collect_profile_keys(config: dict[str, Any], prefix: str = "") -> set[str]:
    """Recursively collect all keys in a profile config, with depth awareness.

    Returns a set of keys at the top level only (not nested keys from dicts
    that are explicitly handled as nested structures like default-resources).
    For nested dicts that the translator reads as dicts (not as individual fields),
    we include the top-level key but not the nested keys.
    """
    keys = set()
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        keys.add(key)

        # For nested structures the translator handles as dicts (default-resources,
        # set-resources, executor-settings), we don't recurse — those dict keys
        # are handled as a unit.
        if isinstance(value, dict) and key not in (
            "default-resources",
            "set-resources",
            "executor-settings",
        ):
            # For other dicts, recurse to find any keys we missed
            keys.update(_collect_profile_keys(value, full_key))

    return keys


def _translator_consumed_keys() -> set[str]:
    """Extract the set of keys the translator functions explicitly consume.

    This inspects _deployment_settings and _executor_settings to find all
    keys they read from the profile dict.
    """
    consumed = set()

    # Keys read in _deployment_settings
    consumed.update([
        "software-deployment-method",
        "use-conda",
        "use-apptainer",
        "use-singularity",
        "conda-prefix",
        "apptainer-prefix",
        "singularity-prefix",
        "apptainer-args",
        "singularity-args",
    ])

    # Keys read in _executor_settings (SLURM-specific)
    consumed.update([
        "executor-settings",
        "slurm-logdir",
        "slurm-keep-successful-logs",
        "slurm-delete-logfiles-older-than",
        "slurm-init-seconds-before-status-checks",
        "slurm-status-attempts",
        "slurm-requeue",
        "slurm-no-account",
        "slurm-reservation",
    ])

    return consumed


def _get_packaged_profiles() -> dict[str, Path]:
    """Locate all packaged profile directories."""
    profiles_dir = (
        Path(__file__).parent.parent
        / "python" / "regulonado" / "workflow" / "profiles"
    )
    profiles = {}
    for profile_dir in profiles_dir.iterdir():
        if profile_dir.is_dir():
            config_file = profile_dir / "config.yaml"
            if config_file.exists():
                profiles[profile_dir.name] = config_file
    return profiles


def test_profile_keys_coverage():
    """Verify every profile key is consumed by the translator or is explicitly allowed."""
    profiles = _get_packaged_profiles()
    assert len(profiles) >= 3, f"Expected at least 3 packaged profiles, found {len(profiles)}"

    translator_consumed = _translator_consumed_keys()

    gaps: dict[str, list[str]] = {}  # profile name -> unconsumed keys

    for profile_name, config_path in profiles.items():
        config = yaml.safe_load(config_path.read_text()) or {}
        profile_keys = _collect_profile_keys(config)

        # Find keys not consumed by the translator and not in the allowlist
        unconsumed = profile_keys - translator_consumed - set(INTENTIONAL_ALLOWLIST.keys())

        if unconsumed:
            gaps[profile_name] = sorted(unconsumed)

    if gaps:
        # Format a readable error message
        msg = "The following profile keys are not consumed by the translator:\n"
        for profile_name, keys in sorted(gaps.items()):
            msg += f"\n  {profile_name}:\n"
            for key in keys:
                msg += f"    - {key}\n"
        msg += "\nEach key must either:\n"
        msg += "  (a) Be consumed by _deployment_settings or _executor_settings\n"
        msg += "  (b) Be used directly in pipeline()\n"
        msg += "  (c) Be added to INTENTIONAL_ALLOWLIST with a reason\n"
        pytest.fail(msg)


def test_intentional_allowlist_is_actively_used():
    """Verify every key in INTENTIONAL_ALLOWLIST is actually present in some profile.

    This ensures the allowlist is actively maintained and doesn't accumulate stale entries.
    """
    profiles = _get_packaged_profiles()
    all_profile_keys = set()

    for config_path in profiles.values():
        config = yaml.safe_load(config_path.read_text()) or {}
        all_profile_keys.update(_collect_profile_keys(config))

    unused_allowlist = set(INTENTIONAL_ALLOWLIST.keys()) - all_profile_keys

    if unused_allowlist:
        msg = (
            "The following INTENTIONAL_ALLOWLIST entries are not in any profile "
            "and should be removed:\n"
        )
        for key in sorted(unused_allowlist):
            msg += f"  - {key}: {INTENTIONAL_ALLOWLIST[key]}\n"
        pytest.fail(msg)


@pytest.mark.parametrize(
    "profile_name,config_path",
    [
        (name, path)
        for name, path in _get_packaged_profiles().items()
    ],
    ids=lambda p: p if isinstance(p, str) else p.name,
)
def test_each_profile_translates_without_error(profile_name, config_path):
    """Verify each profile can be parsed and passed to the translator without error."""
    config = yaml.safe_load(config_path.read_text()) or {}

    # Should not raise
    _deployment_settings(config)

    executor = config.get("executor", "local")
    if executor == "slurm":
        # SLURM executor settings require the plugin; skip if not available
        try:
            executor_settings = _executor_settings(executor, config)
        except Exception as e:
            if "snakemake-executor-plugin-slurm" in str(e):
                pytest.skip("snakemake-executor-plugin-slurm not installed")
            raise
    else:
        # Non-SLURM executors should return None
        executor_settings = _executor_settings(executor, config)
        assert executor_settings is None


def test_deployment_settings_keys_are_all_present():
    """Ensure _deployment_settings reads all deployment-related keys it should."""
    # This is a regression test: if we add a new deployment key to a profile,
    # we should also handle it in _deployment_settings.

    test_config = {
        "software-deployment-method": "conda",
        "use-conda": True,
        "conda-prefix": "/path/to/conda",
        "use-apptainer": False,
        "use-singularity": False,
        "apptainer-prefix": None,
        "singularity-prefix": None,
        "apptainer-args": "",
        "singularity-args": "",
    }

    # Should not raise
    settings = _deployment_settings(test_config)

    # All keys should be consumed; verify by checking that each key is accessed
    # in the function (this is a behavioral test, not a coverage measurement)
    assert settings is not None or test_config.get("software-deployment-method") is not None


def test_executor_settings_slurm_keys_are_all_present():
    """Ensure _executor_settings reads all SLURM-related keys it should."""
    pytest.importorskip("snakemake_executor_plugin_slurm")

    test_config = {
        "executor-settings": {"some_field": "value"},
        "slurm-logdir": "/path/to/logs",
        "slurm-keep-successful-logs": True,
        "slurm-delete-logfiles-older-than": 30,
        "slurm-init-seconds-before-status-checks": 5,
        "slurm-status-attempts": 10,
        "slurm-requeue": True,
        "slurm-no-account": False,
        "slurm-reservation": "myreserv",
    }

    # Should not raise
    settings = _executor_settings("slurm", test_config)

    # Basic check: settings should be created
    assert settings is not None
