"""The genome registry file shared with SeqNado."""

from __future__ import annotations

import json

import pytest
from regulonado.config.genomes import (
    describe_registry,
    genome_config_path,
    load_genome_registry,
)


def _write_registry(root, payload) -> None:
    """Write ``<root>/.config/seqnado/genome_config.json``."""
    path = root / ".config" / "seqnado" / "genome_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))


def test_missing_registry_returns_an_empty_mapping(tmp_path):
    assert load_genome_registry(tmp_path / "nope.json") == {}


def test_invalid_json_returns_an_empty_mapping_without_raising(tmp_path):
    path = tmp_path / "genome_config.json"
    path.write_text("{ this is not json")

    assert load_genome_registry(path) == {}


def test_non_object_json_returns_an_empty_mapping(tmp_path):
    path = tmp_path / "genome_config.json"
    path.write_text(json.dumps(["hg38"]))

    assert load_genome_registry(path) == {}


def test_placeholder_entries_are_skipped(tmp_path):
    path = tmp_path / "genome_config.json"
    path.write_text(
        json.dumps(
            {
                # SeqNado's template stub, written on a fresh install.
                "hg38": {
                    "fasta": "PATH_TO_FASTA",
                    "bt2_index": "PATH_TO_BOWTIE2_INDEX",
                    "chromosome_sizes": "PATH_TO_CHROMOSOME_SIZES",
                },
                "mm39": {
                    "fasta": "/data/mm39/mm39.fa",
                    "bt2_index": "/data/mm39/bt2/mm39",
                    "chromosome_sizes": "/data/mm39/mm39.chrom.sizes",
                    "gtf": "NA",
                },
            }
        )
    )

    genomes = load_genome_registry(path)

    assert set(genomes) == {"mm39"}
    assert genomes["mm39"].name == "mm39"
    assert genomes["mm39"].fasta == "/data/mm39/mm39.fa"
    # 'NA' is SeqNado's not-set marker, not a path.
    assert genomes["mm39"].gtf is None
    assert genomes["mm39"].is_placeholder is False


def test_genome_config_path_honours_seqnado_config(monkeypatch, tmp_path):
    monkeypatch.setenv("SEQNADO_CONFIG", str(tmp_path))

    assert genome_config_path() == tmp_path / ".config" / "seqnado" / "genome_config.json"


def test_registry_is_read_from_seqnado_config_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("SEQNADO_CONFIG", str(tmp_path))
    _write_registry(tmp_path, {"hg38": {"fasta": "/data/hg38/hg38.fa"}})

    genomes = load_genome_registry()

    assert set(genomes) == {"hg38"}
    assert "hg38" in describe_registry()


def test_describe_registry_reports_an_empty_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("SEQNADO_CONFIG", str(tmp_path / "empty"))

    summary = describe_registry()

    assert "no genomes configured" in summary
    assert str(tmp_path / "empty") in summary


@pytest.mark.parametrize("value", [{"fasta": 5}, "not-a-dict"])
def test_unparseable_entries_are_skipped_not_fatal(tmp_path, value):
    path = tmp_path / "genome_config.json"
    path.write_text(json.dumps({"broken": value, "hg38": {"fasta": "/data/hg38.fa"}}))

    genomes = load_genome_registry(path)

    assert set(genomes) == {"hg38"}
