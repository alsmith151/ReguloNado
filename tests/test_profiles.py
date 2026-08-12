"""Snakemake execution preset discovery, shared with SeqNado."""

from __future__ import annotations

from pathlib import Path

import pytest
from regulonado.cli.profiles import (
    format_available_presets,
    get_preset_profiles,
    profile_shortcode,
    resolve_profile_path,
)


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    """Point ``Path.home()`` at a tmp dir so the developer's real config is ignored."""
    home = tmp_path / "home"
    (home / ".config" / "snakemake").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("profile_local_environment", "le"),
        ("profile_slurm_singularity_gpu", "ssg"),
        ("profile_slurm_gpu", "sg"),
        ("profile_aws", "a"),
        # Not a preset directory at all.
        ("slurm_gpu", None),
        ("config.yaml", None),
    ],
)
def test_profile_shortcode(name, expected):
    assert profile_shortcode(name) == expected


def test_get_preset_profiles_includes_regulonados_own():
    profiles = get_preset_profiles()

    assert profiles["sg"] == "profile_slurm_gpu"
    assert profiles["ssg"] == "profile_slurm_singularity_gpu"


def test_format_available_presets_lists_regulonados_own():
    listing = format_available_presets()

    assert "sg (profile_slurm_gpu)" in listing
    assert "ssg (profile_slurm_singularity_gpu)" in listing


def test_resolve_profile_path_returns_a_usable_profile_directory(fake_home):
    path = resolve_profile_path("sg")

    assert path is not None
    assert path.is_dir()
    assert (path / "config.yaml").is_file()


def test_resolve_profile_path_is_case_insensitive(fake_home):
    assert resolve_profile_path("SG") == resolve_profile_path("sg")


def test_resolve_profile_path_returns_none_for_an_unknown_shortcode(fake_home):
    assert resolve_profile_path("not-a-preset") is None
    assert resolve_profile_path(None) is None
    assert resolve_profile_path("") is None


def test_user_installed_profile_wins_over_the_packaged_copy(fake_home):
    packaged = resolve_profile_path("sg")

    user_profile = fake_home / ".config" / "snakemake" / "profile_slurm_gpu"
    user_profile.mkdir(parents=True)
    (user_profile / "config.yaml").write_text("jobs: 1\n")

    resolved = resolve_profile_path("sg")

    assert resolved == user_profile
    assert resolved != packaged
