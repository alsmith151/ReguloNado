"""Genome sequence access and one-hot encoding, shared by counting and training.

Ported from ``unique_enhancer_finding.modelling.sequence``: sequence is read
lazily from the genome FASTA via ``pyfaidx`` rather than materialised
up-front, so callers can hold a region set that is far larger than the
sequence any one of them touches. ``pyfaidx`` reads the standard ``.fai``
index, so a shared, read-only reference that already has one opens as-is.

Coordinates are 0-based half-open throughout: a region ``(start, end)``
covers ``end - start`` bases, ``fasta[chrom][start:end]``.

The one-hot row order (``A0 C1 G2 T3``, an all-zero column for anything
outside ``ACGT``) matches ``regulonado.model.wrapper.one_hot_sequence_tokens``,
so encoded windows from here and from the tokenised training path are
interchangeable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BASE_ORDER",
    "Genome",
    "fetch_sequence",
    "fetch_window",
    "one_hot_encode",
    "open_genome",
]

#: Row order of the one-hot encoding. Matches ``one_hot_sequence_tokens``'s
#: token convention (A0 C1 G2 T3).
BASE_ORDER = "ACGT"
_BASE_INDEX = {base: index for index, base in enumerate(BASE_ORDER)}


class Genome:
    """A ``pyfaidx.Fasta`` handle, opened lazily and re-opened per process.

    A pyfaidx handle (an open file with its own read position) must never be
    shared across forked or spawned worker processes. Rather than requiring
    every caller to remember that, this class defers opening until first use
    and re-opens whenever the current pid differs from the one that opened
    it — the pattern ``RegionTargetDataset._genome`` uses in
    ``unique_enhancer_finding.modelling.dataset``. A single :class:`Genome`
    can therefore be constructed once in the main process and handed to
    ``DataLoader`` workers or a ``ProcessPoolExecutor`` unchanged.
    """

    def __init__(self, path: str | Path, build_index: bool = True) -> None:
        self._path = Path(path).expanduser()
        self._build_index = build_index
        self._fasta: Any = None
        self._pid: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def fasta(self) -> Any:
        """The underlying ``pyfaidx.Fasta``, opened (or re-opened) for the current process.

        An existing ``.fai`` is used as-is, never rebuilt, so a reference directory we can't
        write to still opens; one is built only when missing (and *build_index* allows it).
        """
        if self._fasta is None or self._pid != os.getpid():
            import pyfaidx  # deferred: only needed when reading sequence

            self._fasta = pyfaidx.Fasta(
                str(self._path),
                as_raw=True,
                rebuild=False,
                build_index=self._build_index,
            )
            self._pid = os.getpid()
        return self._fasta

    def contig_length(self, chrom: str) -> int:
        """Length in bp of *chrom*."""
        return len(self.fasta[chrom])

    def __getstate__(self) -> dict[str, Any]:
        # Dropped on pickling (fork/spawn into a worker process): the next
        # access to `.fasta` in that process re-opens its own handle.
        state = self.__dict__.copy()
        state["_fasta"] = None
        state["_pid"] = None
        return state


def open_genome(path: str | Path, build_index: bool = True) -> Genome:
    """Open the genome FASTA at *path* as a lazy, pid-checked :class:`Genome`."""
    return Genome(path, build_index=build_index)


def fetch_sequence(genome: Genome, chrom: str, start: int, end: int) -> str:
    """Fetch the upper-cased sequence over 0-based half-open ``[start, end)`` on ``chrom``.

    ``fasta[chrom][start:end]`` already uses plain Python slicing semantics
    -- 0-based, half-open -- so no offset translation is needed (unlike
    ``pyfaidx.Fasta.get_seq``, which takes 1-based inclusive intervals and is
    deliberately not used here).

    Raises:
        ValueError: if ``[start, end)`` is not a valid non-empty interval.
    """
    if start < 0 or end <= start:
        msg = f"invalid 0-based half-open interval [{start}, {end})"
        raise ValueError(msg)
    return str(genome.fasta[chrom][start:end]).upper()


def one_hot_encode(sequence: str) -> np.ndarray:
    """One-hot encode ``sequence`` to ``[4, len(sequence)]``, rows ``A, C, G, T``.

    Case-insensitive (soft-masked lowercase bases are treated as their
    upper-case base). Any base outside ``ACGT`` -- most commonly ``N``, but
    also IUPAC ambiguity codes -- gets an all-zero column: it does not vote
    for any base, rather than being assigned an arbitrary one.
    """
    codes = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)
    encoded = np.zeros((4, codes.size), dtype=np.float32)
    for base, index in _BASE_INDEX.items():
        encoded[index] = codes == ord(base)
    return encoded


def fetch_window(genome: Genome, chrom: str, start: int, end: int) -> np.ndarray:
    """One-hot encode ``[start, end)`` on ``chrom``, padding past contig ends with zeros.

    ``start`` may be negative and ``end`` may run past the contig length;
    the part of the window outside ``[0, contig_length)`` is filled with
    all-zero columns rather than raising, matching the padding convention
    ``one_hot_sequence_tokens`` uses for out-of-contig bases (bases 4 and
    above -- N and padding -- get no set bit).

    Raises:
        ValueError: if ``end <= start``.
    """
    if end <= start:
        raise ValueError(f"invalid 0-based half-open interval [{start}, {end})")
    length = genome.contig_length(chrom)
    clipped_start = max(start, 0)
    clipped_end = min(end, length)

    encoded = np.zeros((4, end - start), dtype=np.float32)
    if clipped_end > clipped_start:
        sequence = fetch_sequence(genome, chrom, clipped_start, clipped_end)
        offset = clipped_start - start
        encoded[:, offset : offset + (clipped_end - clipped_start)] = one_hot_encode(sequence)
    return encoded
