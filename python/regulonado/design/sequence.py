"""Sequence plumbing for the design module: one-hot arrays and seed resolution.

The search operates on ``(4, context_length)`` one-hot arrays taken from the *dataset window*
that contains each candidate enhancer — the same context the folds were trained and evaluated
on — rather than re-centring on the candidate. :func:`resolve_seeds` is the entry point: it
turns a user-supplied candidate BED into :class:`Seed` objects that carry that context plus the
candidate's editable span in both context and predicted-bin coordinates.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from regulonado.inference import _BASE_LUT, Window, one_hot_context

__all__ = [
    "DatasetWindowIndex",
    "Seed",
    "context_bp_to_pred_bins",
    "decode",
    "fetch_context",
    "one_hot",
    "resolve_seeds",
    "reverse_complement",
    "splice",
]


# --------------------------------------------------------------------------- #
# One-hot arrays                                                             #
# --------------------------------------------------------------------------- #
def one_hot(seq: str) -> np.ndarray:
    """Encode a DNA string as ``(4, L)`` int8. Unknown bases (incl. N) -> all-zero column."""
    codes = _BASE_LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
    out = np.zeros((4, len(seq)), dtype=np.int8)
    valid = codes >= 0
    out[codes[valid], np.nonzero(valid)[0]] = 1
    return out


_DECODE_LUT = np.array(list("ACGT"))


def decode(one_hot_array: np.ndarray) -> str:
    """Decode a ``(4, L)`` one-hot array back to a DNA string; all-zero columns -> 'N'."""
    has_base = one_hot_array.any(axis=0)
    indices = one_hot_array.argmax(axis=0)
    chars = np.where(has_base, _DECODE_LUT[indices], "N")
    return "".join(chars.tolist())


def reverse_complement(one_hot_array: np.ndarray) -> np.ndarray:
    """Reverse-complement a ``(4, L)`` one-hot array (matches ``dataset/build.py``'s convention)."""
    return np.flip(one_hot_array, axis=(0, 1)).copy()


def splice(context: np.ndarray, insert: np.ndarray, start: int) -> np.ndarray:
    """Return a copy of ``context`` with ``insert`` written at columns ``[start, start+L)``."""
    out = context.copy()
    out[:, start : start + insert.shape[1]] = insert
    return out


def fetch_context(
    fasta, chrom: str, center: int, context_length: int, chrom_length: int
) -> np.ndarray:
    """One-hot encode ``context_length`` bp of context centred on ``center``.

    Thin wrapper over ``inference.Window`` + ``inference.one_hot_context``, which already
    zero-pads off chromosome ends.
    """
    ctx_start = center - context_length // 2
    window = Window(
        chrom=chrom,
        pred_start=center,
        pred_end=center,
        ctx_start=ctx_start,
        ctx_end=ctx_start + context_length,
    )
    return one_hot_context(fasta, window, context_length, chrom_length)


# --------------------------------------------------------------------------- #
# Crop mapping                                                                #
# --------------------------------------------------------------------------- #
def context_bp_to_pred_bins(
    start: int, end: int, *, context_length: int, n_pred_bins: int, bin_size: int
) -> slice:
    """Map a ``[start, end)`` span in context coordinates to predicted-bin coordinates.

    Raises if the span falls (even partially) outside the predicted crop — outside it the model
    emits nothing to optimise against.
    """
    crop_bp = (context_length - n_pred_bins * bin_size) // 2
    lo = (start - crop_bp) // bin_size
    hi = -(-(end - crop_bp) // bin_size)  # ceil
    if lo < 0 or hi > n_pred_bins:
        raise ValueError(
            f"Span [{start},{end}) in context coordinates falls outside the predicted crop "
            f"[{crop_bp},{crop_bp + n_pred_bins * bin_size}) (context_length={context_length}, "
            f"n_pred_bins={n_pred_bins}, bin_size={bin_size})"
        )
    return slice(lo, hi)


# --------------------------------------------------------------------------- #
# Seed resolution — candidate BED -> dataset window                          #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Seed:
    """One candidate enhancer resolved against the dataset window that contains it."""

    name: str  # BED col 4 if present, else chrom:start-end
    chrom: str
    cand_start: int  # candidate enhancer, genomic
    cand_end: int
    window: Window  # the dataset window it lives in
    fold_label: str | None  # dataset BED col 4 (train/valid/test split)
    editable: slice  # candidate mapped into context coordinates
    bins: slice  # candidate mapped into predicted-bin coordinates


@dataclass(slots=True)
class _IndexedWindow:
    window: Window
    fold_label: str
    row_index: int


class DatasetWindowIndex:
    """Per-chromosome sorted index over the build-time interval BED.

    Dataset windows are reconstructed exactly as the builder does it: centre on the interval
    midpoint, predicted region = centre ± ``n_pred_bins*bin_size/2``, context = centre ±
    ``context_length/2`` (the same rule as ``dataset/build.py:_signal_intervals`` and
    ``inference.py:iter_windows``).
    """

    def __init__(
        self,
        by_chrom: dict[str, list[_IndexedWindow]],
        *,
        context_length: int,
        n_pred_bins: int,
        bin_size: int,
    ) -> None:
        self._by_chrom = by_chrom
        self.context_length = context_length
        self.n_pred_bins = n_pred_bins
        self.bin_size = bin_size

    @classmethod
    def from_bed(
        cls,
        intervals_bed: str | Path,
        *,
        context_length: int,
        n_pred_bins: int,
        bin_size: int,
    ) -> "DatasetWindowIndex":
        from regulonado.dataset.build import _load_bed_rows

        pred_bp = n_pred_bins * bin_size
        by_chrom: dict[str, list[_IndexedWindow]] = {}
        for row_index, (chrom, start, end, fold) in enumerate(_load_bed_rows(intervals_bed)):
            center = (start + end) // 2
            pred_start = center - pred_bp // 2
            ctx_start = center - context_length // 2
            window = Window(
                chrom=chrom,
                pred_start=pred_start,
                pred_end=pred_start + pred_bp,
                ctx_start=ctx_start,
                ctx_end=ctx_start + context_length,
            )
            by_chrom.setdefault(chrom, []).append(_IndexedWindow(window, fold, row_index))

        for windows in by_chrom.values():
            windows.sort(key=lambda iw: iw.window.pred_start)

        return cls(
            by_chrom, context_length=context_length, n_pred_bins=n_pred_bins, bin_size=bin_size
        )

    def containing(self, chrom: str, start: int, end: int) -> list[_IndexedWindow]:
        """Windows whose predicted region entirely contains ``[start, end)``."""
        windows = self._by_chrom.get(chrom)
        if not windows:
            return []
        pred_bp = self.n_pred_bins * self.bin_size
        starts = [iw.window.pred_start for iw in windows]
        # Any window containing `start` has pred_start in [start - pred_bp, start].
        lo = bisect.bisect_left(starts, start - pred_bp)
        hi = bisect.bisect_right(starts, start)
        return [
            iw
            for iw in windows[lo:hi]
            if iw.window.pred_start <= start and end <= iw.window.pred_end
        ]


def _pick_most_centered(candidates: list[_IndexedWindow], start: int, end: int) -> _IndexedWindow:
    """Most-centred window (maximises the minimum distance to either predicted-region edge).

    Ties break on BED order (smaller row_index in the intervals BED).
    """

    def score(iw: _IndexedWindow) -> tuple[int, int]:
        distance = min(start - iw.window.pred_start, iw.window.pred_end - end)
        return (-distance, iw.row_index)

    return min(candidates, key=score)


def resolve_seeds(
    candidates_bed: str | Path,
    index: DatasetWindowIndex,
    *,
    on_missing: Literal["error", "center", "skip"] = "error",
    pad: int = 0,
) -> list[Seed]:
    """Resolve a candidate BED against the dataset window index.

    A candidate must lie entirely inside a window's predicted region. Multiple containing
    windows pick the most centred one. With no containing window: ``on_missing="error"``
    (default) raises listing every offending candidate; ``"center"`` falls back to a synthetic
    window centred on the candidate; ``"skip"`` drops it.
    """
    from loguru import logger

    from regulonado.dataset.build import _load_bed_rows

    rows = _load_bed_rows(candidates_bed)
    seeds: list[Seed] = []
    missing: list[tuple[str, str, int, int]] = []

    for chrom, start, end, name in rows:
        name = name or f"{chrom}:{start}-{end}"
        candidates = index.containing(chrom, start, end)
        if not candidates:
            missing.append((name, chrom, start, end))
            continue
        best = _pick_most_centered(candidates, start, end)
        if best.fold_label == "train":
            logger.warning(
                f"Candidate {name!r} ({chrom}:{start}-{end}) matched a 'train'-fold dataset "
                f"window — the folds have memorised this window, so specificity gains here are "
                f"the least trustworthy."
            )
        seeds.append(
            _build_seed(name, chrom, start, end, best.window, best.fold_label, index, pad)
        )

    if missing:
        if on_missing == "error":
            listing = ", ".join(
                f"{name} ({chrom}:{start}-{end})" for name, chrom, start, end in missing
            )
            raise ValueError(
                f"{len(missing)} candidate(s) matched no dataset window's predicted region: "
                f"{listing}. Pass on_missing='center' or 'skip' to handle them differently."
            )
        if on_missing == "skip":
            pass
        elif on_missing == "center":
            for name, chrom, start, end in missing:
                center = (start + end) // 2
                pred_bp = index.n_pred_bins * index.bin_size
                pred_start = center - pred_bp // 2
                ctx_start = center - index.context_length // 2
                window = Window(
                    chrom=chrom,
                    pred_start=pred_start,
                    pred_end=pred_start + pred_bp,
                    ctx_start=ctx_start,
                    ctx_end=ctx_start + index.context_length,
                )
                seeds.append(_build_seed(name, chrom, start, end, window, None, index, pad))
        else:
            raise ValueError(f"Unknown on_missing={on_missing!r}")

    return seeds


def _build_seed(
    name: str,
    chrom: str,
    start: int,
    end: int,
    window: Window,
    fold_label: str | None,
    index: DatasetWindowIndex,
    pad: int,
) -> Seed:
    bins = context_bp_to_pred_bins(
        start - window.ctx_start,
        end - window.ctx_start,
        context_length=index.context_length,
        n_pred_bins=index.n_pred_bins,
        bin_size=index.bin_size,
    )
    editable_start = max(0, start - window.ctx_start - pad)
    editable_stop = min(index.context_length, end - window.ctx_start + pad)
    return Seed(
        name=name,
        chrom=chrom,
        cand_start=start,
        cand_end=end,
        window=window,
        fold_label=fold_label,
        editable=slice(editable_start, editable_stop),
        bins=bins,
    )
