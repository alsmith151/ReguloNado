"""Per-nucleotide ISM attribution against one output track, and core-region calling.

This is the *pre-selection* front end for :mod:`regulonado.design`: it finds the sub-span of a
candidate that actually drives a chosen track (in practice the nucleosome-free core), so that
``regulonado design`` optimises — and synthesis pays for — only that span rather than the whole
user-supplied candidate.

The contrast with ``design.search.ism_greedy`` is the point: that function computes the same
per-(position, base) landscape but discards it after picking edits, because it is an optimiser.
Here the landscape *is* the output and nothing is mutated permanently.

Note on resolution: the 32 bp prediction bin is not a limit on the attribution profile. The
profile is indexed by the *mutated position*, so it is base-resolution whatever ``bin_size`` is;
``bins`` only selects which scalar is read back out. What the readout window controls is contrast
— a single-base effect confined to two or three bins is diluted when averaged over a whole
candidate's worth of bins, which is why ``reduction`` is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

from regulonado.design.search import _to_numpy
from regulonado.design.sequence import Seed

__all__ = [
    "AttributionRecord",
    "CoreRegion",
    "IsmResult",
    "TrackReadout",
    "call_cores",
    "ism_scan",
    "merge_attribution_bigwig",
    "write_attributions",
]

BASES = "ACGT"


# --------------------------------------------------------------------------- #
# Readout — one named track, reduced to a scalar                              #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TrackReadout:
    """Scalar readout of a single track over a bin window, ensembled across folds.

    ``__call__`` returns ``(scores (B,), per_fold_scores (F, B))`` — the per-fold array is kept so
    callers can check that folds agree about a core rather than trusting their mean.
    """

    ensemble: Any  # design.predictor.FoldEnsemble, or any object with .predict()
    track_index: int
    bins: slice
    reduction: Literal["mean", "topk", "max"] = "mean"
    topk_bins: int = 10
    fold_reduction: Literal["mean", "median"] = "mean"

    def __call__(self, one_hot_batch) -> tuple[np.ndarray, np.ndarray]:
        preds = self.ensemble.predict(one_hot_batch)  # (F, B, T, N)
        # _to_numpy carries the bf16 -> float32 guard flashzoi needs (numpy has no bfloat16).
        windowed = _to_numpy(preds[:, :, self.track_index, self.bins])  # (F, B, W)

        if self.reduction == "mean":
            per_fold = windowed.mean(axis=-1)
        elif self.reduction == "max":
            per_fold = windowed.max(axis=-1)
        elif self.reduction == "topk":
            k = min(self.topk_bins, windowed.shape[-1])
            per_fold = np.sort(windowed, axis=-1)[..., -k:].mean(axis=-1)
        else:
            raise ValueError(f"Unknown reduction {self.reduction!r}")

        if self.fold_reduction == "mean":
            scores = per_fold.mean(axis=0)
        elif self.fold_reduction == "median":
            scores = np.median(per_fold, axis=0)
        else:
            raise ValueError(f"Unknown fold_reduction {self.fold_reduction!r}")
        return scores, per_fold


# --------------------------------------------------------------------------- #
# ISM sweep                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class IsmResult:
    """One saturation-mutagenesis sweep over a candidate's editable span.

    ``importance`` is positive where mutating *lowers* the track — i.e. where the base matters.
    ``effect`` keeps the opposite (conventional ISM) sign, ``alt - ref``, so it reads as a normal
    red/blue mutation-effect matrix. Both are indexed relative to ``editable.start``, and both
    hold ``NaN`` at positions the sweep skipped.
    """

    ref_score: float
    per_fold_ref: np.ndarray  # (F,)
    effect: np.ndarray  # (4, L) — alt - ref, ref base's own row 0.0
    importance: np.ndarray  # (L,) — mean over alts of (ref - alt)
    per_fold_importance: np.ndarray  # (F, L)
    positions: np.ndarray  # (n_scanned,) context coordinates
    editable: slice
    ref_bases: np.ndarray  # (L,) ref base index, -1 where the context has no base (N)


def ism_scan(
    readout: TrackReadout,
    seed: Seed,
    context: np.ndarray,
    *,
    positions: Sequence[int] | None = None,
    stride: int = 1,
    batch_size: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> IsmResult:
    """Score every alternative base at every (subsampled) editable position, once.

    ``positions`` restricts the sweep to explicit context coordinates (e.g. resolved from a motif
    BED); otherwise every ``stride``-th editable position is scanned. Semantics match
    ``design.search.ism_greedy`` so the two commands agree on what a position is.
    """
    editable = seed.editable
    width = editable.stop - editable.start
    if positions is not None:
        wanted = sorted(set(int(p) for p in positions))
        scan = [p for p in wanted if editable.start <= p < editable.stop]
    else:
        scan = list(range(editable.start, editable.stop, stride))
    if not scan:
        raise ValueError(
            f"No positions to scan for candidate {seed.name!r}: editable span "
            f"[{editable.start},{editable.stop}) yielded nothing at stride={stride}"
        )

    ref_scores, ref_per_fold = readout(context[None])
    ref_score = float(ref_scores[0])
    per_fold_ref = np.asarray(ref_per_fold)[:, 0]
    n_folds = per_fold_ref.shape[0]

    # A column is a reference base only if it is a *clean* one-hot. An all-zero column (an N) or
    # a multi-hot column has no identity edit, so all four substitutions are real and all four
    # are scored. This test is taken from tangermeme's saturation_mutagenesis, which gets it
    # right where a bare argmax would silently treat an N as an 'A'.
    ref_bases = np.full(width, -1, dtype=np.int8)
    for position in scan:
        column = context[:, position]
        if int((column == 1).sum()) == 1 and int(column.sum()) == 1:
            ref_bases[position - editable.start] = int(column.argmax())

    # Descriptors only, contexts materialised per chunk — the same pattern as ism_greedy, and the
    # reason this is not delegated to tangermeme.saturation_mutagenesis: that function builds
    # every mutant up front (`X[i].repeat(n_edits, 1, 1)`), which at Borzoi's 524 288 bp context
    # is 3.8 GB for a 600 bp candidate and 12.6 GB for a 2 kb one, against ~17 MB resident here.
    # It also takes an nn.Module and has no notion of a fold ensemble.
    proposals: list[tuple[int, int]] = []
    for position in scan:
        ref_base = ref_bases[position - editable.start]
        for base_index in range(4):
            if base_index != ref_base:
                proposals.append((position, base_index))

    alt = np.full((4, width), np.nan)  # alt - ref, alt rows only
    per_fold_alt = np.full((n_folds, 4, width), np.nan)

    total = len(proposals)
    for start in range(0, total, batch_size):
        chunk = proposals[start : start + batch_size]
        arrays = []
        for position, base_index in chunk:
            mutated = context.copy()
            mutated[:, position] = 0
            mutated[base_index, position] = 1
            arrays.append(mutated)
        scores, per_fold_scores = readout(np.stack(arrays))
        for offset, (position, base_index) in enumerate(chunk):
            column = position - editable.start
            alt[base_index, column] = scores[offset] - ref_score
            per_fold_alt[:, base_index, column] = per_fold_scores[:, offset] - per_fold_ref
        if on_progress is not None:
            on_progress(min(start + batch_size, total), total)

    scanned = np.array([p - editable.start for p in scan], dtype=int)
    importance = np.full(width, np.nan)
    per_fold_importance = np.full((n_folds, width), np.nan)
    # Only reduce over scanned columns: an all-NaN column would warn and yield NaN anyway.
    importance[scanned] = -np.nanmean(alt[:, scanned], axis=0)
    per_fold_importance[:, scanned] = -np.nanmean(per_fold_alt[:, :, scanned], axis=1)

    effect = alt.copy()
    for column in scanned:
        ref_base = ref_bases[column]
        if ref_base >= 0:
            effect[ref_base, column] = 0.0

    return IsmResult(
        ref_score=ref_score,
        per_fold_ref=per_fold_ref,
        effect=effect,
        importance=importance,
        per_fold_importance=per_fold_importance,
        positions=np.array(scan, dtype=int),
        editable=editable,
        ref_bases=ref_bases,
    )


# --------------------------------------------------------------------------- #
# Core-region calling                                                         #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CoreRegion:
    """A called high-attribution segment, in context coordinates."""

    start: int
    end: int
    score: float  # mean smoothed importance inside the segment
    zscore: float  # (score - candidate mean) / candidate sd
    peak: int  # argmax of the smoothed profile
    anchor: int  # centroid or peak, whichever --anchor selected
    rank: int  # 0 = best
    clamped: bool = False  # fixed-width expansion hit the window bound
    fixed_width: bool = False

    @property
    def width(self) -> int:
        return self.end - self.start


def _smooth(importance: np.ndarray, smooth_bp: int) -> np.ndarray:
    """Centred, NaN-aware rolling mean.

    pandas handles the three things a hand-rolled version gets wrong: NaN holes left by
    ``stride > 1``, the array edges, and leaving an all-NaN stretch as NaN rather than 0.
    """
    import pandas as pd

    window = max(1, int(smooth_bp))
    return (
        pd.Series(importance).rolling(window, center=True, min_periods=1).mean().to_numpy()
    )


def call_cores(
    importance: np.ndarray,
    *,
    editable: slice,
    smooth_bp: int = 25,
    quantile: float = 0.90,
    min_width_bp: int = 50,
    merge_gap_bp: int = 20,
    min_zscore: float = 1.5,
    max_cores: int = 1,
    anchor: Literal["centroid", "peak"] = "centroid",
    fix_width: int | None = None,
    bounds: tuple[int, int] | None = None,
) -> tuple[list[CoreRegion], dict]:
    """Call high-attribution core segments from a per-base importance profile.

    Smooth -> threshold at ``quantile`` -> bridge sub-``merge_gap_bp`` dips -> drop runs shorter
    than ``min_width_bp`` -> rank by mean smoothed importance. Returns the ranked cores (in
    context coordinates) and a diagnostics dict recording why, if at all, nothing was called.

    ``min_zscore`` is the guard that matters for pre-selection: on a flat profile a quantile
    threshold *always* fires, so a segment must also sit that many standard deviations above the
    candidate's own mean before it is worth synthesising. Failing it returns no cores rather than
    a spurious one.
    """
    from scipy.ndimage import (
        binary_closing,
        center_of_mass,
        find_objects,
        label,
        labeled_comprehension,
    )

    width = importance.shape[0]
    smoothed = _smooth(importance, smooth_bp)
    diagnostics: dict[str, Any] = {
        "threshold": float("nan"),
        "baseline": float("nan"),
        "scale": float("nan"),
        "n_segments": 0,
        "core_called": False,
        "reason": "",
    }

    if not np.isfinite(smoothed).any():
        diagnostics["reason"] = "no_finite_positions"
        return [], diagnostics

    baseline = float(np.nanmean(smoothed))
    scale = float(np.nanstd(smoothed))
    diagnostics["baseline"] = baseline
    diagnostics["scale"] = scale

    def _finalize(cores: list[CoreRegion]) -> list[CoreRegion]:
        if fix_width:
            for core in cores:
                _apply_fix_width(core, fix_width, editable, bounds)
        return cores

    # A candidate narrower than the minimum width has no interior to choose from.
    if width < min_width_bp:
        peak = int(editable.start + np.nanargmax(smoothed))
        diagnostics.update(
            core_called=True, reason="candidate_shorter_than_min_width", n_segments=1
        )
        return _finalize(
            [
                CoreRegion(
                    start=editable.start,
                    end=editable.stop,
                    score=baseline,
                    zscore=0.0,
                    peak=peak,
                    anchor=peak,
                    rank=0,
                )
            ]
        ), diagnostics

    threshold = float(np.nanquantile(smoothed, quantile))
    diagnostics["threshold"] = threshold
    mask = np.isfinite(smoothed) & (smoothed >= threshold)

    if merge_gap_bp >= 1:
        # structure of size k bridges gaps of size < k, so k = gap + 1 merges gaps <= gap.
        # border_value=1 is essential: the default erodes segments touching the array edge,
        # which would silently drop a core sitting at the candidate boundary.
        mask = binary_closing(
            mask, structure=np.ones(int(merge_gap_bp) + 1, dtype=bool), border_value=1
        )

    labels, n_labels = label(mask)
    if n_labels == 0:
        diagnostics["reason"] = "no_positions_above_threshold"
        return [], diagnostics

    slices = find_objects(labels)
    indices = np.arange(1, n_labels + 1)
    means = labeled_comprehension(smoothed, labels, indices, np.nanmean, float, np.nan)
    # center_of_mass needs non-negative weights; importance may legitimately go negative.
    weights = np.clip(np.nan_to_num(smoothed, nan=0.0), 0.0, None)
    centroids = center_of_mass(weights, labels, indices)

    # A centred box filter of width w ramps a step edge over w bp, so thresholding the smoothed
    # profile cuts roughly w/2 off each end of a real segment. Undo that erosion, or every core
    # is reported ~smooth_bp narrower than it is (an 80 bp block comes back as 56 bp at the
    # default smooth_bp=25).
    grow = int(smooth_bp) // 2

    candidates: list[CoreRegion] = []
    for i, (segment, mean) in enumerate(zip(slices, means)):
        # Undo the erosion *before* the width test, so --min-width-bp means the width of the core
        # actually emitted. Testing the eroded span instead would reject every segment narrower
        # than min_width_bp + smooth_bp.
        span = segment[0]
        # Cap the correction at the segment's own width, so a feature much narrower than
        # smooth_bp (which is smeared, not eroded) cannot be inflated out of all proportion.
        step = min(grow, span.stop - span.start)
        span = slice(max(0, span.start - step), min(width, span.stop + step))
        if span.stop - span.start < min_width_bp:
            continue
        local = smoothed[span]
        peak = int(span.start + np.nanargmax(local))
        centroid = centroids[i][0] if n_labels > 1 else centroids[0]
        centroid = float(np.atleast_1d(centroid)[0])
        if not np.isfinite(centroid):
            centroid = (span.start + span.stop) / 2.0
        zscore = (float(mean) - baseline) / scale if scale > 0 else 0.0
        anchor_index = int(round(centroid)) if anchor == "centroid" else peak
        candidates.append(
            CoreRegion(
                start=editable.start + span.start,
                end=editable.start + span.stop,
                score=float(mean),
                zscore=zscore,
                peak=editable.start + peak,
                anchor=editable.start + anchor_index,
                rank=0,
            )
        )

    diagnostics["n_segments"] = len(candidates)
    if not candidates:
        diagnostics["reason"] = "no_segment_meets_min_width"
        return [], diagnostics

    candidates.sort(key=lambda core: core.score, reverse=True)
    if candidates[0].zscore < min_zscore:
        diagnostics["reason"] = "low_zscore"
        return [], diagnostics

    kept = [core for core in candidates if core.zscore >= min_zscore][:max_cores]
    for rank, core in enumerate(kept):
        core.rank = rank
    diagnostics.update(core_called=True, reason="ok")
    return _finalize(kept), diagnostics


def _apply_fix_width(
    core: CoreRegion, fix_width: int, editable: slice, bounds: tuple[int, int] | None
) -> None:
    """Re-centre ``core`` on its anchor, sized to exactly ``fix_width`` bp inside ``bounds``."""
    lo, hi = bounds if bounds is not None else (editable.start, editable.stop)
    if hi - lo < fix_width:
        core.start, core.end = lo, hi
        core.clamped = True
        core.fixed_width = False
        return
    start = core.anchor - fix_width // 2
    start = max(lo, min(start, hi - fix_width))
    core.clamped = start != core.anchor - fix_width // 2
    core.start, core.end = start, start + fix_width
    core.fixed_width = True


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class AttributionRecord:
    """One candidate's sweep, called cores and diagnostics."""

    seed: Seed
    ism: IsmResult
    smoothed: np.ndarray
    cores: list[CoreRegion]
    diagnostics: dict = field(default_factory=dict)


def _to_genomic(record: AttributionRecord, context_position: int) -> int:
    return record.seed.window.ctx_start + context_position


def write_attributions(
    out_dir: str | Path,
    records: list[AttributionRecord],
    *,
    run_info: dict,
    chrom_sizes: dict[str, int] | None = None,
    bigwig: bool = True,
    rtol: float = 0.01,
) -> None:
    """Write ``core_regions.bed``, ``cores.tsv``, ``summary.tsv``, ``attributions.tsv``,
    ``attributions.bw`` and ``run.json``.

    Called after every candidate so a long run is resumable/inspectable mid-flight, matching
    ``design.report.write_designs``.
    """
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _core_frame(records).to_csv(out_dir / "cores.tsv", sep="\t", index=False)
    _bed_frame(records).to_csv(
        out_dir / "core_regions.bed", sep="\t", index=False, header=False
    )
    _summary_frame(records).to_csv(out_dir / "summary.tsv", sep="\t", index=False)
    _attribution_frame(records).to_csv(out_dir / "attributions.tsv", sep="\t", index=False)
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2, default=str))

    if bigwig and chrom_sizes:
        _write_attribution_bigwig(out_dir, records, chrom_sizes, rtol)


def _core_frame(records: list[AttributionRecord]):
    import pandas as pd

    rows = []
    for record in records:
        seed = record.seed
        for core in record.cores:
            rows.append(
                {
                    "name": f"{seed.name}_core{core.rank}",
                    "candidate": seed.name,
                    "chrom": seed.chrom,
                    "start": _to_genomic(record, core.start),
                    "end": _to_genomic(record, core.end),
                    "width": core.width,
                    "rank": core.rank,
                    "score": core.score,
                    "zscore": core.zscore,
                    "peak": _to_genomic(record, core.peak),
                    "anchor": _to_genomic(record, core.anchor),
                    "fixed_width": core.fixed_width,
                    "clamped": core.clamped,
                    "candidate_start": seed.cand_start,
                    "candidate_end": seed.cand_end,
                    "fold_label": seed.fold_label or "",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "name", "candidate", "chrom", "start", "end", "width", "rank", "score",
            "zscore", "peak", "anchor", "fixed_width", "clamped", "candidate_start",
            "candidate_end", "fold_label",
        ],
    )


def _bed_frame(records: list[AttributionRecord]):
    """BED6, directly consumable as ``regulonado design --candidates``."""
    import pandas as pd

    cores = _core_frame(records)
    if cores.empty:
        return pd.DataFrame(columns=["chrom", "start", "end", "name", "score", "strand"])
    # BED score column is a 0-1000 integer; rank z-scores onto that range for browser display.
    z = cores["zscore"].to_numpy(dtype=float)
    span = np.nanmax(z) - np.nanmin(z) if len(z) > 1 else 0.0
    scaled = (
        np.full(len(z), 1000)
        if span <= 0
        else np.clip(((z - np.nanmin(z)) / span * 1000).round(), 0, 1000)
    )
    bed = cores[["chrom", "start", "end", "name"]].copy()
    bed["score"] = scaled.astype(int)
    bed["strand"] = "."
    return bed


def _summary_frame(records: list[AttributionRecord]):
    import pandas as pd

    rows = []
    for record in records:
        seed, ism, diag = record.seed, record.ism, record.diagnostics
        best = record.cores[0] if record.cores else None
        candidate_width = seed.cand_end - seed.cand_start
        rows.append(
            {
                "name": seed.name,
                "chrom": seed.chrom,
                "start": seed.cand_start,
                "end": seed.cand_end,
                "fold_label": seed.fold_label or "",
                "ref_score": ism.ref_score,
                "per_fold_ref": ",".join(f"{v:.6g}" for v in ism.per_fold_ref.tolist()),
                "n_positions_scanned": int(ism.positions.size),
                "threshold": diag.get("threshold", float("nan")),
                "baseline": diag.get("baseline", float("nan")),
                "scale": diag.get("scale", float("nan")),
                "n_segments": diag.get("n_segments", 0),
                "core_called": bool(diag.get("core_called", False)),
                "reason": diag.get("reason", ""),
                "core_chrom": seed.chrom if best else "",
                "core_start": _to_genomic(record, best.start) if best else "",
                "core_end": _to_genomic(record, best.end) if best else "",
                "core_width": best.width if best else "",
                "core_zscore": best.zscore if best else "",
                "core_fraction_of_candidate": (
                    best.width / candidate_width if best and candidate_width else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _attribution_frame(records: list[AttributionRecord]):
    """Long form, one row per scanned position. The delta_* columns carry the full 4xL matrix."""
    import pandas as pd

    frames = []
    for record in records:
        seed, ism = record.seed, record.ism
        columns = ism.positions - ism.editable.start
        in_core = np.zeros(ism.importance.shape[0], dtype=bool)
        for core in record.cores:
            in_core[core.start - ism.editable.start : core.end - ism.editable.start] = True
        ref_bases = ism.ref_bases[columns]
        frame = pd.DataFrame(
            {
                "name": seed.name,
                "chrom": seed.chrom,
                "position": ism.positions + seed.window.ctx_start,
                "ref_base": [BASES[b] if b >= 0 else "N" for b in ref_bases],
                "importance": ism.importance[columns],
                "importance_smooth": record.smoothed[columns],
                "in_core": in_core[columns],
            }
        )
        for base_index, base in enumerate(BASES):
            frame[f"delta_{base}"] = ism.effect[base_index, columns]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "name", "chrom", "position", "ref_base", "importance", "importance_smooth",
                "in_core", "delta_A", "delta_C", "delta_G", "delta_T",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _write_attribution_bigwig(
    out_dir: Path,
    records: list[AttributionRecord],
    chrom_sizes: dict[str, int],
    rtol: float,
) -> None:
    """Per-base importance track, reusing inference's collapse + pybigtools writer."""
    import pandas as pd

    from regulonado.inference import _write_bigwigs, collapse_bins

    intervals: list[tuple[str, int, int, float]] = []
    for record in records:
        seed, ism = record.seed, record.ism
        chrom_length = chrom_sizes.get(seed.chrom)
        if chrom_length is None:
            continue
        intervals.extend(
            collapse_bins(
                np.nan_to_num(ism.importance, nan=0.0),
                seed.chrom,
                seed.window.ctx_start + ism.editable.start,
                1,
                rtol,
                chrom_length,
            )
        )
    if not intervals:
        return

    # Candidates sharing a dataset window can overlap; pybigtools rejects overlapping intervals.
    frame = pd.DataFrame(intervals, columns=["chrom", "start", "end", "value"])
    rank = {name: i for i, name in enumerate(chrom_sizes)}
    frame = frame.sort_values(
        by=["chrom", "start"], key=lambda col: col.map(rank) if col.name == "chrom" else col
    ).reset_index(drop=True)
    prior_end = frame.groupby("chrom")["end"].cummax().shift(fill_value=-1)
    same_chrom = frame["chrom"].eq(frame["chrom"].shift())
    overlapping = same_chrom & (frame["start"] < prior_end)
    if overlapping.any():
        from loguru import logger

        logger.warning(
            f"{int(overlapping.sum())} overlapping interval(s) dropped from attributions.bw — "
            f"candidates whose scanned spans overlap keep only the first one's profile in the "
            f"overlap. The per-candidate values in attributions.tsv are unaffected."
        )
    frame = frame[~overlapping]

    _write_bigwigs(
        out_dir,
        ["attributions"],
        [0],
        {0: list(frame.itertuples(index=False, name=None))},
        chrom_sizes,
    )


def merge_attribution_bigwig(
    attributions_tsv: str | Path, out_path: str | Path, *, fasta_path: str | Path
) -> Path:
    """Rebuild a single BigWig from a merged ``attributions.tsv``.

    Used by the workflow's merge step: shards hold disjoint candidates but can still overlap
    genomically, so the per-shard BigWigs cannot simply be concatenated. Chromosome sizes come
    from the FASTA's ``.fai`` index.
    """
    import pandas as pd

    from regulonado.inference import _write_bigwigs, collapse_bins, read_chrom_sizes

    out_path = Path(out_path)
    chrom_sizes = read_chrom_sizes(Path(f"{fasta_path}.fai"))

    frame = pd.read_csv(attributions_tsv, sep="\t")
    intervals: list[tuple[str, int, int, float]] = []
    if not frame.empty:
        for (name, chrom), group in frame.groupby(["name", "chrom"], sort=False):
            chrom_length = chrom_sizes.get(str(chrom))
            if chrom_length is None:
                continue
            group = group.sort_values("position")
            intervals.extend(
                collapse_bins(
                    np.nan_to_num(group["importance"].to_numpy(dtype=float), nan=0.0),
                    str(chrom),
                    int(group["position"].iloc[0]),
                    1,
                    0.01,
                    chrom_length,
                )
            )

    if not intervals:
        # pybigtools still needs a valid header, so emit an empty-but-well-formed file.
        import pybigtools

        pybigtools.open(str(out_path), "w").write(chrom_sizes, iter([]))
        return out_path

    frame = pd.DataFrame(intervals, columns=["chrom", "start", "end", "value"])
    rank = {chrom: i for i, chrom in enumerate(chrom_sizes)}
    frame = frame.sort_values(
        by=["chrom", "start"], key=lambda col: col.map(rank) if col.name == "chrom" else col
    ).reset_index(drop=True)
    prior_end = frame.groupby("chrom")["end"].cummax().shift(fill_value=-1)
    same_chrom = frame["chrom"].eq(frame["chrom"].shift())
    frame = frame[~(same_chrom & (frame["start"] < prior_end))]

    written = _write_bigwigs(
        out_path.parent,
        [out_path.stem],
        [0],
        {0: list(frame.itertuples(index=False, name=None))},
        chrom_sizes,
    )
    return written[0]
