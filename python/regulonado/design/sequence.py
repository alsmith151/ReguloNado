"""Sequence plumbing for the design module: seed resolution against dataset windows.

The search operates on ``(4, context_length)`` one-hot arrays taken from the *dataset window*
that contains each candidate enhancer — the same context the folds were trained and evaluated
on — rather than re-centring on the candidate. :func:`resolve_seeds` is the entry point: it
turns a user-supplied candidate BED into :class:`Seed` objects that carry that context plus the
candidate's editable span in both context and predicted-bin coordinates.

One-hot encode/decode helpers (``one_hot``, ``decode``, ``reverse_complement``) live in
:mod:`regulonado.genomics`.

:func:`apply_neutral_flanks` replaces the genomic flank around a candidate with synthetic
background, for simulating an MPRA reporter construct where the designed element is not at its
native locus. Every mode other than ``"genomic"`` is **out of distribution** for a model trained
only on endogenous genomic sequence: the model has never seen a shuffled or uniform flank during
training, so its predictions there carry that caveat. ``"dinuc-shuffle"`` is the least-bad of the
non-genomic choices, since it perturbs motif content while leaving base composition and CpG
statistics untouched. ``"uniform"`` is the worst: a 0.25-everywhere column is a value the model's
one-hot input channel has literally never taken during training.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from regulonado.genomics import Window, decode, one_hot, one_hot_context, window_for_interval

logger = logging.getLogger(__name__)
_WARNED_NEUTRAL_FLANK_MODES: set[str] = set()

__all__ = [
    "DatasetWindowIndex",
    "Seed",
    "apply_neutral_flanks",
    "context_bp_to_pred_bins",
    "fetch_context",
    "resolve_seeds",
    "splice",
]


def splice(context: np.ndarray, insert: np.ndarray, start: int) -> np.ndarray:
    """Return a copy of ``context`` with ``insert`` written at columns ``[start, start+L)``."""
    out = context.copy()
    out[:, start : start + insert.shape[1]] = insert
    return out


def fetch_context(
    fasta, chrom: str, center: int, context_length: int, chrom_length: int
) -> np.ndarray:
    """One-hot encode ``context_length`` bp of context centred on ``center``.

    Thin wrapper over ``genomics.Window`` + ``genomics.one_hot_context``, which already
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
# Neutral flanks                                                             #
# --------------------------------------------------------------------------- #
def apply_neutral_flanks(
    context: np.ndarray,
    keep: slice,
    *,
    mode: Literal["genomic", "shuffle", "dinuc-shuffle", "uniform"],
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace everything outside ``keep`` with neutral background.

    ``context`` is a ``(4, L)`` one-hot array in the ``genomics.one_hot`` convention (rows are
    A/C/G/T in that order, columns are positions; an unrecognised base is an all-zero column).
    ``keep`` is a column slice (step 1) that is left untouched -- typically the candidate's
    editable span, so its flanks (everything before ``keep.start`` and after ``keep.stop``,
    clamped to the array) are what gets replaced. The two flanks are pooled and shuffled
    *together* as one background, not independently, since they represent a single synthetic
    context rather than two.

    Always returns a new array; ``context`` is reused across search rounds by callers and must
    never be mutated under them.

    Modes:
      - ``"genomic"``: identity -- returns ``context`` unchanged (but as a copy).
      - ``"shuffle"``: permutes the flank columns among themselves. Exactly preserves
        mononucleotide (per-channel) composition; does not preserve dinucleotide frequencies.
      - ``"dinuc-shuffle"``: Altschul-Erikson dinucleotide-preserving shuffle of the flank
        sequence (see :func:`_dinucleotide_shuffle`). Preserves both mononucleotide composition
        and dinucleotide counts, so CpG content is retained.
      - ``"uniform"``: writes 0.25 into every channel of every flank column. Since 0.25 is not
        representable in the input's integer dtype, this mode returns a ``float32`` array even
        when ``context`` is ``int8``; the ``keep`` region's values are preserved exactly (just
        promoted to float), so it stays numerically identical.

    Columns outside ``keep`` that are not a clean one-hot column (all-zero, i.e. an ``N``; or
    already fractional, e.g. a re-applied ``"uniform"`` flank) are left untouched *in place* by
    ``"shuffle"`` and ``"dinuc-shuffle"`` -- only genuine one-hot columns are pooled and
    permuted among each other. This keeps both shuffles well-defined (there is no dinucleotide
    graph edge to build through an ``N``) without silently corrupting or "inventing" a base for
    positions the caller did not give one.
    """
    length = context.shape[1]
    start, stop, step = keep.indices(length)
    if step != 1:
        raise ValueError(f"`keep` must have step 1, got step={step}")
    start = max(0, min(start, length))
    stop = max(start, min(stop, length))
    flank_idx = np.concatenate([np.arange(0, start), np.arange(stop, length)])

    if mode == "genomic" or flank_idx.size == 0:
        return context.copy()

    if mode not in _WARNED_NEUTRAL_FLANK_MODES:
        logger.warning(
            "flank_mode=%r replaces endogenous genomic context with out-of-distribution "
            "synthetic sequence; treat absolute scores as diagnostic", mode
        )
        _WARNED_NEUTRAL_FLANK_MODES.add(mode)

    if mode == "uniform":
        out = context.astype(np.float32, copy=True)
        out[:, flank_idx] = 0.25
        return out

    if mode not in ("shuffle", "dinuc-shuffle"):
        raise ValueError(f"Unknown mode {mode!r}")

    flank_cols = context[:, flank_idx]
    is_clean_one_hot = (flank_cols.sum(axis=0) == 1) & np.isin(flank_cols, (0, 1)).all(axis=0)
    clean_idx = flank_idx[is_clean_one_hot]

    out = context.copy()
    if clean_idx.size < 2:
        return out  # nothing meaningful to shuffle (0 or 1 clean columns)

    if mode == "shuffle":
        perm = rng.permutation(clean_idx.size)
        out[:, clean_idx] = context[:, clean_idx][:, perm]
        return out

    seq = decode(context[:, clean_idx])
    shuffled = _dinucleotide_shuffle(seq, rng)
    out[:, clean_idx] = one_hot(shuffled).astype(context.dtype)
    return out


def _dinucleotide_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Altschul-Erikson dinucleotide-preserving shuffle of ``seq`` (bases drawn from A/C/G/T).

    Standard construction (Altschul & Erikson 1985): build the doublet graph -- one directed
    edge per consecutive pair in ``seq``, from the first base to the second. Fix, for every
    nucleotide other than ``seq[-1]``, one random outgoing edge as its "last edge"; this defines
    a candidate spanning structure rooted at ``seq[-1]``. Retry that random draw until every
    nucleotide can reach ``seq[-1]`` by following last-edge pointers -- this is exactly the
    condition for an Eulerian path over the doublet graph ending at ``seq[-1]`` to exist (an
    Eulerian path must exit through each node's last-remaining edge last, so if that edge can't
    reach the end, the path can't either). With connectivity guaranteed, randomly reorder each
    node's *other* edges, append its last edge at the end of its list, and walk the path from
    ``seq[0]``. Every edge (i.e. every original dinucleotide occurrence) is consumed exactly
    once, so dinucleotide counts -- and therefore mononucleotide counts -- are exactly preserved;
    only their order changes.
    """
    if len(seq) < 3:
        return seq  # 0 or 1 dinucleotide edges: no freedom, order is already forced

    nucleotides = sorted(set(seq))
    last_ch = seq[-1]

    while True:
        # Recomputed fresh each attempt: a rejected draw must not leak state into the retry.
        dinuc_next: dict[str, list[str]] = {x: [] for x in nucleotides}
        for a, b in zip(seq, seq[1:]):
            dinuc_next[a].append(b)

        last_edge: dict[str, str] = {}
        remaining = {x: list(ys) for x, ys in dinuc_next.items()}
        for x in nucleotides:
            if x == last_ch:
                continue
            # Every occurrence of x other than seq[-1] itself is followed by something, and
            # x != last_ch means x never *is* seq[-1], so this list is always non-empty.
            choices = remaining[x]
            last_edge[x] = choices.pop(int(rng.integers(len(choices))))

        if _all_reach_last(last_edge, nucleotides, last_ch):
            break

    # Rebuild the per-nucleotide edge lists, pull out exactly one instance of each node's fixed
    # last edge, shuffle what remains, then put the last edge back at the end of each list --
    # so a walk that drains each list front-to-back uses every node's last edge last.
    edge_lists: dict[str, list[str]] = {x: [] for x in nucleotides}
    for a, b in zip(seq, seq[1:]):
        edge_lists[a].append(b)
    for x, y in last_edge.items():
        edge_lists[x].remove(y)
    for x in nucleotides:
        edges = edge_lists[x]
        for i in range(len(edges) - 1, 0, -1):  # Fisher-Yates
            j = int(rng.integers(i + 1))
            edges[i], edges[j] = edges[j], edges[i]
    for x, y in last_edge.items():
        edge_lists[x].append(y)

    out = [seq[0]]
    prev = seq[0]
    for _ in range(len(seq) - 2):
        nxt = edge_lists[prev].pop(0)
        out.append(nxt)
        prev = nxt
    out.append(seq[-1])
    return "".join(out)


def _all_reach_last(last_edge: dict[str, str], nucleotides: list[str], last_ch: str) -> bool:
    """True iff every nucleotide reaches ``last_ch`` by following ``last_edge`` pointers."""
    reached = {last_ch}
    changed = True
    while changed:
        changed = False
        for x, y in last_edge.items():
            if y in reached and x not in reached:
                reached.add(x)
                changed = True
    return all(x in reached for x in nucleotides)


# --------------------------------------------------------------------------- #
# Crop mapping                                                                #
# --------------------------------------------------------------------------- #
def context_bp_to_pred_bins(
    start: int,
    end: int,
    *,
    context_length: int,
    n_pred_bins: int,
    bin_size: int,
    crop_bp: int | None = None,
) -> slice:
    """Map a ``[start, end)`` span in context coordinates to predicted-bin coordinates.

    ``crop_bp`` is the number of context bp preceding the predicted crop. When ``None`` (the
    default) it is derived as ``(context_length - n_pred_bins*bin_size) // 2`` — the assumption
    that the predicted crop is centred in the context. That assumption does not hold for every
    architecture (e.g. region-count heads built on ``TrunkWindow``, see
    ``training/regions/live.py``, whose scored crop is ``first_bin * bin_size`` and not
    necessarily the centred value); callers with the true crop should pass it explicitly.

    Raises if the span falls (even partially) outside the predicted crop — outside it the model
    emits nothing to optimise against.
    """
    if crop_bp is None:
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
    ``context_length/2`` — see ``genomics.window_for_interval``, the rule shared with
    ``inference.py:iter_windows``.
    """

    def __init__(
        self,
        by_chrom: dict[str, list[_IndexedWindow]],
        *,
        context_length: int,
        n_pred_bins: int,
        bin_size: int,
        crop_bp: int | None = None,
        snap_bp: int | None = None,
    ) -> None:
        self._by_chrom = by_chrom
        self.context_length = context_length
        self.n_pred_bins = n_pred_bins
        self.bin_size = bin_size
        self.crop_bp = crop_bp
        self.snap_bp = snap_bp
        # Windows are not assumed to share a width (see `containing`) — track the widest one
        # actually present so the bisect bound stays correct if that ever stops being true.
        self._max_width = max(
            (
                iw.window.pred_end - iw.window.pred_start
                for windows in by_chrom.values()
                for iw in windows
            ),
            default=n_pred_bins * bin_size,
        )

    @classmethod
    def from_bed(
        cls,
        intervals_bed: str | Path,
        *,
        context_length: int,
        n_pred_bins: int,
        bin_size: int,
        crop_bp: int | None = None,
        snap_bp: int | None = None,
    ) -> "DatasetWindowIndex":
        from regulonado.genomics import read_intervals

        frame = read_intervals(intervals_bed)
        if "name" in frame.columns:
            rows = frame[["chrom", "start", "end", "name"]].itertuples(index=False, name=None)
        else:
            rows = (
                (chrom, start, end, "")
                for chrom, start, end in frame[["chrom", "start", "end"]].itertuples(
                    index=False, name=None
                )
            )

        by_chrom: dict[str, list[_IndexedWindow]] = {}
        for row_index, (chrom, start, end, fold) in enumerate(rows):
            chrom, start, end, fold = str(chrom), int(start), int(end), str(fold)
            window = window_for_interval(
                chrom,
                start,
                end,
                context_length=context_length,
                n_pred_bins=n_pred_bins,
                bin_size=bin_size,
                crop_bp=crop_bp,
                snap_bp=snap_bp,
            )
            by_chrom.setdefault(chrom, []).append(_IndexedWindow(window, fold, row_index))

        for windows in by_chrom.values():
            windows.sort(key=lambda iw: iw.window.pred_start)

        return cls(
            by_chrom,
            context_length=context_length,
            n_pred_bins=n_pred_bins,
            bin_size=bin_size,
            crop_bp=crop_bp,
            snap_bp=snap_bp,
        )

    @classmethod
    def from_region_table(
        cls,
        regions: str | Path,
        *,
        context_length: int,
        n_pred_bins: int,
        bin_size: int,
        crop_bp: int | None = None,
        snap_bp: int | None = None,
    ) -> "DatasetWindowIndex":
        """Build the index from a region-count dataset's region table.

        Region-count datasets (``RegionCountData``, see ``counts/dataset.py``) specify their
        regions as a parquet with ``chrom``/``start``/``end`` columns, usually accompanied by
        ``target_start``/``target_end`` (the scored sub-span within the region — preferred over
        ``start``/``end`` when present, matching ``normalization.py::read_regions``'s column
        conventions) and a ``split`` column (used here as the per-interval ``fold_label``, same
        role as a BED's 4th column in :meth:`from_bed`, so the train-fold "memorised window"
        warning in :func:`resolve_seeds` still fires).

        Falls back to :meth:`from_bed` (via ``genomics.read_intervals``) for a non-``.parquet``
        path, so BED-shaped region tables keep working unchanged.

        ``crop_bp``/``snap_bp`` are forwarded to ``genomics.window_for_interval`` unchanged —
        see that function's docstring for the distinction between them.
        """
        path = Path(regions)
        if path.suffix.lower() != ".parquet":
            return cls.from_bed(
                path,
                context_length=context_length,
                n_pred_bins=n_pred_bins,
                bin_size=bin_size,
                crop_bp=crop_bp,
                snap_bp=snap_bp,
            )

        import polars as pl

        frame = pl.read_parquet(path)
        columns = set(frame.columns)
        start_col = "target_start" if "target_start" in columns else "start"
        end_col = "target_end" if "target_end" in columns else "end"
        has_split = "split" in columns

        select_cols = ["chrom", start_col, end_col]
        if has_split:
            select_cols.append("split")
        rows = frame.select(select_cols).iter_rows()

        by_chrom: dict[str, list[_IndexedWindow]] = {}
        for row_index, row in enumerate(rows):
            if has_split:
                chrom, start, end, fold = row
                fold = "" if fold is None else str(fold)
            else:
                chrom, start, end = row
                fold = ""
            chrom, start, end = str(chrom), int(start), int(end)
            window = window_for_interval(
                chrom,
                start,
                end,
                context_length=context_length,
                n_pred_bins=n_pred_bins,
                bin_size=bin_size,
                crop_bp=crop_bp,
                snap_bp=snap_bp,
            )
            by_chrom.setdefault(chrom, []).append(_IndexedWindow(window, fold, row_index))

        for windows in by_chrom.values():
            windows.sort(key=lambda iw: iw.window.pred_start)

        return cls(
            by_chrom,
            context_length=context_length,
            n_pred_bins=n_pred_bins,
            bin_size=bin_size,
            crop_bp=crop_bp,
            snap_bp=snap_bp,
        )

    def containing(self, chrom: str, start: int, end: int) -> list[_IndexedWindow]:
        """Windows whose predicted region entirely contains ``[start, end)``.

        Coordinates are 0-based, half-open (BED convention): a window with predicted region
        ``[pred_start, pred_end)`` contains ``[start, end)`` iff ``pred_start <= start`` and
        ``end <= pred_end``. Windows are not assumed to share a width. The candidate bracket is
        found by bisecting on ``pred_start`` alone using ``self._max_width`` (the widest window
        actually in the index): any containing window must have ``pred_start <= start`` (or it
        starts after the candidate), and since its width is at most ``self._max_width``, it must
        also have ``pred_start >= end - self._max_width`` (or even at its widest it would end
        before ``end``). Every window in that bracket is then exactly filtered on its own
        ``pred_end``, so this is correct regardless of per-window width — the bisect only narrows
        the scan, it never substitutes for the real containment check.
        """
        windows = self._by_chrom.get(chrom)
        if not windows:
            return []
        starts = [iw.window.pred_start for iw in windows]
        lo = bisect.bisect_left(starts, end - self._max_width)
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
    score_pad_bp: int = 0,
) -> list[Seed]:
    """Resolve a candidate BED against the dataset window index.

    A candidate must lie entirely inside a window's predicted region. Multiple containing
    windows pick the most centred one. With no containing window: ``on_missing="error"``
    (default) raises listing every offending candidate; ``"center"`` falls back to a synthetic
    window centred on the candidate; ``"skip"`` drops it.

    ``pad`` widens the *editable* span the search may mutate; ``score_pad_bp`` widens only the
    *scored* bins (e.g. to capture a nucleosome-free-region dip flanking a narrow candidate)
    without letting the search edit that flanking sequence.
    """
    from regulonado.genomics import read_intervals

    _frame = read_intervals(candidates_bed)
    if "name" in _frame.columns:
        rows = [
            (str(chrom), int(start), int(end), str(name))
            for chrom, start, end, name in _frame[
                ["chrom", "start", "end", "name"]
            ].itertuples(index=False, name=None)
        ]
    else:
        rows = [
            (str(chrom), int(start), int(end), "")
            for chrom, start, end in _frame[["chrom", "start", "end"]].itertuples(
                index=False, name=None
            )
        ]
    seeds: list[Seed] = []
    missing: list[tuple[str, str, int, int]] = []

    for chrom, start, end, name in rows:
        # "." is BED's own placeholder for "no value" (as in GTF/GFF); a literal dot is
        # not a usable name any more than an empty column 4 is.
        name = name if name and name != "." else f"{chrom}:{start}-{end}"
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
            _build_seed(
                name, chrom, start, end, best.window, best.fold_label, index, pad, score_pad_bp
            )
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
                window = window_for_interval(
                    chrom,
                    start,
                    end,
                    context_length=index.context_length,
                    n_pred_bins=index.n_pred_bins,
                    bin_size=index.bin_size,
                    crop_bp=index.crop_bp,
                    snap_bp=index.snap_bp,
                )
                seeds.append(
                    _build_seed(name, chrom, start, end, window, None, index, pad, score_pad_bp)
                )
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
    score_pad_bp: int,
) -> Seed:
    bins = context_bp_to_pred_bins(
        start - window.ctx_start - score_pad_bp,
        end - window.ctx_start + score_pad_bp,
        context_length=index.context_length,
        n_pred_bins=index.n_pred_bins,
        bin_size=index.bin_size,
        crop_bp=index.crop_bp,
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
