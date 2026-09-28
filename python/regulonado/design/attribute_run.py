"""Run entry point for the ISM-attribution stage: ``AttributionConfig -> AttributionResult``.

Split out of the CLI so that ``regulonado attribute`` and the workflow rule share exactly one
implementation and one validated config (:class:`regulonado.config.models.AttributionConfig`),
instead of the CLI re-deriving its own ad hoc option parsing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from regulonado.config.models import AttributionConfig
    from regulonado.design.attribution import AttributionRecord
    from regulonado.design.predictor import FoldEnsemble
    from regulonado.design.sequence import Seed

logger = logging.getLogger(__name__)

__all__ = ["AttributionResult", "run_attribution"]


@dataclass(slots=True)
class AttributionResult:
    """Output paths and summary counts from one :func:`run_attribution` call."""

    out_dir: Path
    core_regions_bed: Path
    n_candidates: int
    n_cores_called: int
    n_cores_total: int


def _load_ensemble(config: "AttributionConfig") -> "FoldEnsemble":
    from regulonado.design.predictor import FoldSpec, load_fold_ensemble

    if not config.checkpoint_dirs:
        raise ValueError(
            "attribution.checkpoint_dirs must list at least one checkpoint for a standalone run"
        )
    logger.info(f"Loading {len(config.checkpoint_dirs)} fold(s)...")
    dataset_dir = Path(config.dataset_dir) if config.dataset_dir else None
    return load_fold_ensemble(
        [FoldSpec(checkpoint_dir=Path(c), dataset_dir=dataset_dir) for c in config.checkpoint_dirs],
        device=config.device,
        batch_size=config.batch_size,
        mode=config.fold_mode,
        model_kind=config.model_kind,
    )


def _check_region_settings(config: "AttributionConfig", ensemble) -> None:
    """Authoritative re-check of ``AttributionConfig._region_settings_are_compatible``.

    The config-time validator can only fire when ``model_kind`` was given explicitly (under
    ``"auto"`` the checkpoint isn't readable at config-validation time); this runs unconditionally
    once the ensemble is loaded and its real kind is known, so a region-count checkpoint loaded
    under ``model_kind="auto"`` still gets caught before wasting a scan on settings that make no
    sense for it.
    """
    from regulonado.design.region_predictor import RegionFoldEnsemble

    if not isinstance(ensemble, RegionFoldEnsemble):
        return
    if config.bin_reduction != "mean":
        raise ValueError(
            f"{ensemble} is a region-count ensemble, which requires bin_reduction='mean'; got "
            f"{config.bin_reduction!r}"
        )
    if "topk_bins" in config.model_fields_set:
        raise ValueError(
            "This is a region-count ensemble, which has no bin window for topk_bins to select "
            "within; leave it unset"
        )


def _resolve_track_indices(config: "AttributionConfig", ensemble: "FoldEnsemble") -> list[int]:
    target = config.targets[0]
    if target.track is not None:
        from regulonado.inference import resolve_tracks

        indices = resolve_tracks([target.track], ensemble.track_names)
        logger.info(
            f"Attributing against track {ensemble.track_names[indices[0]]!r} (index {indices[0]})"
        )
        return indices

    from regulonado.design.region_predictor import RegionFoldEnsemble

    if isinstance(ensemble, RegionFoldEnsemble):
        from regulonado.design.objective import group_track_groups

        if target.group_by != "source":
            logger.info(
                f"group_by={target.group_by!r} is ignored for region models: 'target' resolves "
                "directly against the ensemble's group_names"
            )
        groups = group_track_groups(
            ensemble.group_names, target=target.target, exclude_groups=tuple(config.exclude_tracks)
        )
        indices = groups.target_idx.nonzero().flatten().tolist()
        logger.info(
            f"Attributing against group {target.target!r}: "
            f"{len(indices)} region group(s): {[ensemble.group_names[i] for i in indices]}"
        )
        return indices

    from regulonado.design.objective import resolve_track_group_indices

    track_sheet = Path(config.track_sheet) if config.track_sheet else None
    dataset_dir = Path(config.dataset_dir) if config.dataset_dir else None
    indices = resolve_track_group_indices(
        ensemble.track_names,
        group_by=target.group_by,
        target=target.target,
        exclude_tracks=set(config.exclude_tracks),
        track_sheet=track_sheet,
        dataset_dir=dataset_dir,
    )
    logger.info(
        f"Attributing against group {target.group_by}={target.target!r}: "
        f"{len(indices)} track(s): {[ensemble.track_names[i] for i in indices]}"
    )
    return indices


def _resolve_intervals(config: "AttributionConfig") -> Path:
    if config.intervals:
        return Path(config.intervals)
    if not config.dataset_dir:
        raise ValueError(
            "Provide attribution.intervals, or dataset_dir with a tracks.parquet recording "
            "'bed_file'"
        )
    from regulonado.tracks_table import read_track_table

    dataset_dir = Path(config.dataset_dir)
    table = read_track_table(dataset_dir / "tracks.parquet")
    bed_file = table.attrs.get("bed_file")
    if not bed_file:
        raise ValueError(f"No 'bed_file' recorded in {dataset_dir / 'tracks.parquet'}")
    return Path(bed_file)


def _load_scan_positions(config: "AttributionConfig") -> list[tuple[str, int, int]] | None:
    if config.positions is None:
        return None
    from regulonado.genomics import read_intervals

    frame = read_intervals(config.positions)[["chrom", "start", "end"]]
    return [
        (str(chrom), int(start), int(end))
        for chrom, start, end in frame.itertuples(index=False, name=None)
    ]


def _prepare_fasta(config: "AttributionConfig"):
    import pyfaidx

    fasta = pyfaidx.Fasta(str(config.fasta), as_raw=True, sequence_always_upper=False)
    chrom_sizes = {name: len(fasta[name]) for name in fasta.keys()}
    return fasta, chrom_sizes


def _log_projected_passes(seeds: list["Seed"], config: "AttributionConfig") -> None:
    n_folds = len(config.checkpoint_dirs or [])
    if config.method == "gradient":
        logger.info(
            f"Projected forward+backward passes: {len(seeds) * n_folds:,} "
            f"({len(seeds)} candidate(s) x {n_folds} fold(s), 1 pass each)"
        )
        return
    total_positions = sum(
        len(range(seed.editable.start, seed.editable.stop, config.stride)) for seed in seeds
    )
    logger.info(
        f"Projected forward passes: ~{total_positions * 3 * n_folds:,} "
        f"({total_positions} position(s) x 3 alt bases x {n_folds} fold(s))"
    )


def _build_run_info(
    config: "AttributionConfig", ensemble, track_indices: list[int], n_candidates: int
) -> dict:
    target = config.targets[0]
    selector = (
        target.track if target.track is not None else f"group:{target.group_by}={target.target}"
    )
    return {
        "status": "in_progress",
        "candidates": config.candidates,
        "intervals": config.intervals,
        "dataset_dir": config.dataset_dir,
        "checkpoints": list(config.checkpoint_dirs or []),
        "track": selector,
        "track_indices": track_indices,
        "selected_track_names": [ensemble.track_names[i] for i in track_indices],
        "track_names": list(ensemble.track_names),
        "bin_reduction": config.bin_reduction,
        "topk_bins": config.topk_bins,
        "fold_reduction": config.fold_reduction,
        "pad": config.pad,
        "stride": config.stride,
        "smooth_bp": config.smooth_bp,
        "quantile": config.quantile,
        "min_width_bp": config.min_width_bp,
        "merge_gap_bp": config.merge_gap_bp,
        "min_zscore": config.min_zscore,
        "max_cores_per_candidate": config.max_cores_per_candidate,
        "anchor": config.anchor,
        "fix_width": config.fix_width,
        "n_candidates": n_candidates,
        "model_kind": config.model_kind,
        "flank_mode": config.flank_mode,
        "flank_keep": config.flank_keep,
        "flank_keep_bp": config.flank_keep_bp,
        # The fixed constant `_score_one_candidate` seeds np.random.SeedSequence with per
        # candidate index -- see its docstring comment for why this has no config knob.
        "flank_seed": 0,
    }


def _seed_scan_positions(
    seed: "Seed", scan_positions: list[tuple[str, int, int]] | None
) -> list[int] | None:
    if scan_positions is None:
        return None
    return [
        position - seed.window.ctx_start
        for chrom, start, end in scan_positions
        if chrom == seed.chrom
        for position in range(start, end)
    ]


def _flank_keep_span(config: "AttributionConfig", seed: "Seed", ensemble) -> slice:
    """The context-coordinate span :func:`apply_neutral_flanks` leaves untouched.

    ``"candidate"`` is the candidate's own editable span; ``"scored_span"`` is the predicted-crop
    span the readout actually reads (``seed.bins`` mapped back through ``ensemble.crop_bp``/
    ``bin_size``) -- these differ whenever ``pad``/``score_pad_bp``-style widening put the
    editable span and the scored bins at different widths. Both are then widened by
    ``config.flank_keep_bp`` on each side and clamped to the context.
    """
    if config.flank_keep == "candidate":
        base_start, base_stop = seed.editable.start, seed.editable.stop
    else:
        base_start = ensemble.crop_bp + seed.bins.start * ensemble.bin_size
        base_stop = ensemble.crop_bp + seed.bins.stop * ensemble.bin_size
    pad = config.flank_keep_bp
    return slice(max(0, base_start - pad), min(ensemble.context_length, base_stop + pad))


def _score_one_candidate(
    config: "AttributionConfig",
    index: int,
    total: int,
    seed: "Seed",
    ensemble,
    track_indices: list[int],
    fasta,
    chrom_sizes: dict[str, int],
    scan_positions: list[tuple[str, int, int]] | None,
) -> "AttributionRecord":
    from regulonado.design.attribution import (
        AttributionRecord,
        TrackReadout,
        _smooth,
        call_cores,
        grad_scan,
        ism_scan,
    )
    from regulonado.design.region_predictor import RegionContrastReadout, RegionFoldEnsemble
    from regulonado.design.sequence import apply_neutral_flanks
    from regulonado.genomics import one_hot_context

    logger.info(f"[{index}/{total}] {seed.name} ({seed.chrom}:{seed.cand_start}-{seed.cand_end})")
    chrom_length = chrom_sizes.get(seed.chrom)
    if chrom_length is None:
        raise ValueError(f"Chromosome {seed.chrom!r} not present in {config.fasta}")
    context = one_hot_context(fasta, seed.window, ensemble.context_length, chrom_length)

    # A no-op path at the default mode="genomic" -- skipped entirely so profile behaviour stays
    # provably untouched, rather than routed through an identity call to apply_neutral_flanks.
    if config.flank_mode != "genomic":
        # No AttributionConfig.seed knob exists (attribution's scans are otherwise
        # deterministic), so the "run seed" component is a fixed constant here; the candidate
        # index still makes every candidate's background independent and reruns deterministic.
        rng = np.random.default_rng(np.random.SeedSequence([0, index]))
        keep = _flank_keep_span(config, seed, ensemble)
        context = apply_neutral_flanks(context, keep, mode=config.flank_mode, rng=rng)

    if config.method == "gradient":
        result = grad_scan(
            ensemble,
            seed,
            context,
            track_indices=track_indices,
            bins=seed.bins,
            reduction=config.bin_reduction,
            topk_bins=config.topk_bins,
            fold_reduction=config.fold_reduction,
            positions=_seed_scan_positions(seed, scan_positions),
            stride=config.stride,
        )
    else:
        if isinstance(ensemble, RegionFoldEnsemble):
            readout = RegionContrastReadout(
                ensemble, group_index=track_indices[0], fold_reduction=config.fold_reduction
            )
        else:
            readout = TrackReadout(
                ensemble,
                track_indices=track_indices,
                bins=seed.bins,
                reduction=config.bin_reduction,
                topk_bins=config.topk_bins,
                fold_reduction=config.fold_reduction,
            )
        result = ism_scan(
            readout,
            seed,
            context,
            positions=_seed_scan_positions(seed, scan_positions),
            stride=config.stride,
            batch_size=config.batch_size,
        )
    cores, diagnostics = call_cores(
        result.importance,
        editable=seed.editable,
        smooth_bp=config.smooth_bp,
        quantile=config.quantile,
        min_width_bp=config.min_width_bp,
        merge_gap_bp=config.merge_gap_bp,
        min_zscore=config.min_zscore,
        max_cores=config.max_cores_per_candidate,
        anchor=config.anchor,
        fix_width=config.fix_width,
        bounds=(
            seed.window.pred_start - seed.window.ctx_start,
            seed.window.pred_end - seed.window.ctx_start,
        ),
    )
    if not cores:
        logger.warning(f"  no core called for {seed.name}: {diagnostics.get('reason')}")
    else:
        best = cores[0]
        logger.info(
            f"  core {seed.chrom}:{seed.window.ctx_start + best.start}-"
            f"{seed.window.ctx_start + best.end} ({best.width} bp, z={best.zscore:.2f})"
        )
    return AttributionRecord(
        seed=seed,
        ism=result,
        smoothed=_smooth(result.importance, config.smooth_bp),
        cores=cores,
        diagnostics=diagnostics,
    )


def _process_candidates(
    config: "AttributionConfig",
    seeds: list["Seed"],
    ensemble,
    track_indices: list[int],
    fasta,
    chrom_sizes: dict[str, int],
    scan_positions: list[tuple[str, int, int]] | None,
    run_info: dict,
    out_dir: Path,
    call_cores: bool = True,
) -> list["AttributionRecord"]:
    from regulonado.design.attribution import write_attributions

    records = []
    for candidate_index, seed in enumerate(seeds, start=1):
        records.append(
            _score_one_candidate(
                config,
                candidate_index,
                len(seeds),
                seed,
                ensemble,
                track_indices,
                fasta,
                chrom_sizes,
                scan_positions,
            )
        )
        # Checkpoint after every candidate so a long run stays inspectable and resumable.
        write_attributions(
            out_dir,
            records,
            run_info=run_info,
            chrom_sizes=chrom_sizes,
            bigwig=config.bigwig,
            rtol=config.rtol,
            call_cores=call_cores,
        )
    return records


def run_attribution(config: "AttributionConfig", *, call_cores: bool = True) -> AttributionResult:
    """Locate the high-attribution core of each candidate by in-silico mutagenesis.

    Scores every alternative base at every position against one output track (``config.targets``
    must hold exactly one target for a standalone run), then calls the contiguous sub-span that
    drives it. Writes ``core_regions.bed`` plus supporting tables/BigWig to ``config.out_dir``,
    checkpointing after every candidate.
    """
    from regulonado.design.attribution import write_attributions
    from regulonado.design.sequence import DatasetWindowIndex, resolve_seeds

    if not config.fasta:
        raise ValueError("attribution.fasta is required")
    if not config.out_dir:
        raise ValueError("attribution.out_dir is required")
    if len(config.targets) != 1:
        raise ValueError(
            f"run_attribution expects exactly one target; got {len(config.targets)}. A workflow "
            "batch with several targets runs this once per target."
        )
    if config.smooth_bp > 1 and config.stride > config.smooth_bp:
        logger.warning(
            f"stride {config.stride} exceeds smooth_bp {config.smooth_bp}; smoothing windows "
            "will often contain a single scanned position. Raise smooth_bp or lower stride."
        )

    out_dir = Path(config.out_dir)
    intervals = _resolve_intervals(config)
    ensemble = _load_ensemble(config)
    _check_region_settings(config, ensemble)
    track_indices = _resolve_track_indices(config, ensemble)

    from regulonado.design.region_predictor import RegionFoldEnsemble

    index_kwargs = dict(
        context_length=ensemble.context_length,
        n_pred_bins=ensemble.n_pred_bins,
        bin_size=ensemble.bin_size,
        crop_bp=ensemble.crop_bp,
        snap_bp=getattr(ensemble, "snap_bp", None),
    )
    if isinstance(ensemble, RegionFoldEnsemble) and intervals.suffix.lower() == ".parquet":
        index = DatasetWindowIndex.from_region_table(intervals, **index_kwargs)
    else:
        index = DatasetWindowIndex.from_bed(intervals, **index_kwargs)
    seeds = resolve_seeds(config.candidates, index, on_missing=config.on_missing, pad=config.pad)
    logger.info(f"Resolved {len(seeds)} candidate(s) against dataset windows")
    _log_projected_passes(seeds, config)

    scan_positions = _load_scan_positions(config)
    fasta, chrom_sizes = _prepare_fasta(config)

    run_info = _build_run_info(config, ensemble, track_indices, len(seeds))
    records = _process_candidates(
        config,
        seeds,
        ensemble,
        track_indices,
        fasta,
        chrom_sizes,
        scan_positions,
        run_info,
        out_dir,
        call_cores=call_cores,
    )

    n_called = sum(1 for record in records if record.cores)
    n_cores_total = sum(len(record.cores) for record in records)
    run_info["status"] = "complete"
    run_info["n_cores_called"] = n_called
    run_info["n_cores_total"] = n_cores_total
    write_attributions(
        out_dir,
        records,
        run_info=run_info,
        chrom_sizes=chrom_sizes,
        bigwig=config.bigwig,
        rtol=config.rtol,
        call_cores=call_cores,
    )

    return AttributionResult(
        out_dir=out_dir,
        core_regions_bed=out_dir / "core_regions.bed",
        n_candidates=len(records),
        n_cores_called=n_called,
        n_cores_total=n_cores_total,
    )
