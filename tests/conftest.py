"""Shared test helpers for the regulonado test suite.

Consolidates fixtures/helpers that used to be re-declared, near-identically,
in multiple test files (docs/audit-2026-09-11.xml, finding F20). ``tests/``
has no ``__init__.py`` so it is not a package: pytest's default (prepend)
import mode puts this directory on ``sys.path`` for every test module, which
means the plain functions/classes below can be imported directly, e.g.
``from conftest import write_bed``.

Only genuinely identical (or trivially parameterisable) duplicates were
consolidated here; helpers that differed in a way that mattered to the test
were left where they were.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn


class TinyBackbone(nn.Module):
    """Minimal 1x1-conv backbone stub (4 one-hot channels -> 8 features).

    Previously defined identically as ``TinyBackbone`` in
    test_predict_bigwig.py and ``_TinyBackbone`` in test_design.py.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv1d(4, 8, 1)
        self.feature_dim = 8

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(input_ids)

    def iter_named_blocks(self):
        yield "proj", self.proj


def write_bed(path: Path, rows: list[tuple]) -> Path:
    """Write ``rows`` as a tab-separated file with a trailing newline.

    Previously defined identically as ``_write_bed`` in test_attribution.py
    and test_design.py.
    """
    path.write_text("\n".join("\t".join(str(field) for field in row) for row in rows) + "\n")
    return path


def write_bigwig(path: Path, values: list[tuple], size: int = 2000) -> None:
    """Write a tiny single-chromosome (``chr1``) BigWig for tests.

    ``values`` are ``(chrom, start, end, value)`` tuples; they are sorted by
    start position before writing since pybigtools requires sorted input.

    Previously defined as ``_make_bigwig`` (test_track_qc.py,
    test_tracks_cli.py) and ``_write_bigwig`` (test_normalization_anchor.py) —
    equivalent except for the default ``size`` and that the anchor version
    sorted its input; both behaviours are preserved here.
    """
    pybigtools = pytest.importorskip("pybigtools")
    writer = pybigtools.open(str(path), "w")
    writer.write({"chr1": size}, iter(sorted(values, key=lambda v: v[1])))
