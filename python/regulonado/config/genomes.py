"""The genome registry, shared with SeqNado.

SeqNado records where each genome's FASTA, chrom.sizes, GTF and blacklist live in
``~/.config/seqnado/genome_config.json`` (``SEQNADO_CONFIG`` overrides the home
directory). ReguloNado reads the same file so a genome configured once is
available to both tools.

We read the JSON directly rather than calling
``seqnado.config.user_input.load_genome_configs``, which calls ``sys.exit(1)``
when the file is missing — unusable from library code. When SeqNado is installed
its ``GenomeConfig`` model is used to validate entries so the error messages are
SeqNado's own.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger
from pydantic import BaseModel


class GenomeEntry(BaseModel):
    """One genome's resource paths, as recorded by SeqNado."""

    name: str
    fasta: str | None = None
    chromosome_sizes: str | None = None
    gtf: str | None = None
    genes: str | None = None
    blacklist: str | None = None
    bt2_index: str | None = None
    star_index: str | None = None

    @property
    def is_placeholder(self) -> bool:
        """True for the template stub SeqNado writes on a fresh install."""
        return any(
            value and "PATH_TO" in value
            for value in (self.fasta, self.bt2_index, self.chromosome_sizes)
        )


def genome_config_path() -> Path:
    """Location of the shared genome registry."""
    root = Path(os.environ.get("SEQNADO_CONFIG") or Path.home())
    return root / ".config" / "seqnado" / "genome_config.json"


def load_genome_registry(path: str | Path | None = None) -> dict[str, GenomeEntry]:
    """Load the genome registry, or return an empty mapping if there is none.

    Placeholder entries from SeqNado's template are skipped: they contain
    ``PATH_TO`` markers rather than real paths and would only produce confusing
    downstream failures.
    """
    config_path = Path(path) if path is not None else genome_config_path()
    if not config_path.exists():
        logger.debug(f"No genome registry at {config_path}")
        return {}

    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not read genome registry {config_path}: {exc}")
        return {}

    if not isinstance(raw, dict):
        logger.warning(f"Genome registry {config_path} is not a JSON object; ignoring")
        return {}

    genomes: dict[str, GenomeEntry] = {}
    for name, values in raw.items():
        if not isinstance(values, dict):
            continue
        try:
            entry = GenomeEntry(name=name, **{k: v for k, v in values.items() if v != "NA"})
        except Exception as exc:
            logger.debug(f"Skipping genome {name!r}: {exc}")
            continue
        if entry.is_placeholder:
            logger.debug(f"Skipping placeholder genome entry {name!r}")
            continue
        genomes[name] = entry

    return genomes


def describe_registry() -> str:
    """One-line summary for CLI messages."""
    path = genome_config_path()
    genomes = load_genome_registry(path)
    if not genomes:
        return f"no genomes configured ({path})"
    return f"{', '.join(sorted(genomes))} (from {path})"
