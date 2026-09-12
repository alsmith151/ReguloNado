"""Run entry point for the design stage: ``DesignConfig -> DesignResult``.

Split out of the CLI so that ``regulonado design`` and the workflow rule share exactly one
implementation and one validated config (:class:`regulonado.config.models.DesignConfig`),
instead of the CLI re-deriving its own ad hoc option parsing. Mirrors
``design/attribute_run.py``'s split for the attribution stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from regulonado.config.models import DesignConfig, DesignTarget
    from regulonado.design.predictor import FoldEnsemble
    from regulonado.design.search import DesignState
    from regulonado.design.sequence import Seed

logger = logging.getLogger(__name__)

__all__ = ["DesignResult", "run_design"]


@dataclass(slots=True)
class DesignResult:
    """Output paths and summary counts from one :func:`run_design` call."""

    out_dir: Path
    target: str
    method: str
    n_candidates: int


def _trajectory_table(wandb_module, history: list[dict]):
    """Build a wandb.Table of the per-round trajectory, one row per round.

    Column set is the union across rounds (round 0 and non-improving rounds lack "positions";
    AdaLead rounds lack "n_edits"), so every row gets every column, blank where not recorded.
    """
    fixed = [
        "round",
        "sequence",
        "energy",
        "specificity",
        "target",
        "target_gain",
        "offtarget_boost",
        "n_edits",
    ]
    dynamic = sorted(
        {key for entry in history for key in entry if key not in (*fixed, "positions")}
    )
    columns = [*fixed, "positions", *dynamic]
    rows = []
    for entry in history:
        row = []
        for column in columns:
            value = entry.get(column, "")
            if column == "positions" and value:
                value = ",".join(str(p) for p in value)
            row.append(value)
        rows.append(row)
    return wandb_module.Table(columns=columns, data=rows)


def _seed_everything(seed: int | None) -> int:
    """Seed numpy/torch once; candidate-specific generators below derive from this value."""
    run_seed = 0 if seed is None else int(seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    return run_seed


def _load_folds(config: "DesignConfig") -> tuple["FoldEnsemble", "FoldEnsemble | None"]:
    """Load the design fold ensemble and, if configured, the held-out ensemble."""
    from regulonado.design.predictor import FoldEnsemble, FoldSpec

    if not config.checkpoint_dirs:
        raise ValueError(
            "design.checkpoint_dirs must list at least one checkpoint for a standalone run"
        )
    dataset_dir = Path(config.dataset_dir) if config.dataset_dir else None

    logger.info(f"Loading {len(config.checkpoint_dirs)} design fold(s)...")
    design_folds = [
        FoldSpec(checkpoint_dir=Path(c), dataset_dir=dataset_dir) for c in config.checkpoint_dirs
    ]
    ensemble = FoldEnsemble(
        design_folds, device=config.device, batch_size=config.batch_size, mode=config.fold_mode
    )

    if config.holdout_checkpoint is None:
        return ensemble, None

    holdout_ensemble = FoldEnsemble(
        [FoldSpec(checkpoint_dir=Path(config.holdout_checkpoint), dataset_dir=dataset_dir)],
        device=config.device,
        batch_size=config.batch_size,
        mode=config.fold_mode,
    )
    holdout_geometry = (
        holdout_ensemble.context_length,
        holdout_ensemble.n_pred_bins,
        holdout_ensemble.bin_size,
    )
    design_geometry = (ensemble.context_length, ensemble.n_pred_bins, ensemble.bin_size)
    if holdout_geometry != design_geometry:
        raise ValueError(
            f"Held-out fold geometry {holdout_geometry} does not match design fold "
            f"geometry {design_geometry}"
        )
    return ensemble, holdout_ensemble


def _resolve_intervals(config: "DesignConfig") -> Path:
    if config.intervals:
        return Path(config.intervals)
    if not config.dataset_dir:
        raise ValueError(
            "Provide design.intervals, or design.dataset_dir with a tracks.parquet recording "
            "'bed_file'"
        )
    from regulonado.tracks_table import read_track_table

    dataset_dir = Path(config.dataset_dir)
    table = read_track_table(dataset_dir / "tracks.parquet")
    bed_file = table.attrs.get("bed_file")
    if not bed_file:
        raise ValueError(f"No 'bed_file' recorded in {dataset_dir / 'tracks.parquet'}")
    return Path(bed_file)


def _resolve_seeds(
    config: "DesignConfig", ensemble: "FoldEnsemble", intervals: Path
) -> list["Seed"]:
    from regulonado.design.sequence import DatasetWindowIndex, resolve_seeds

    index = DatasetWindowIndex.from_bed(
        intervals,
        context_length=ensemble.context_length,
        n_pred_bins=ensemble.n_pred_bins,
        bin_size=ensemble.bin_size,
    )
    seeds = resolve_seeds(config.candidates, index, on_missing=config.on_missing, pad=config.pad)
    logger.info(f"Resolved {len(seeds)} candidate(s) against dataset windows")
    return seeds


def _resolve_groups(config: "DesignConfig", target: "DesignTarget", ensemble: "FoldEnsemble"):
    from regulonado.design.objective import resolve_track_groups

    track_sheet = Path(config.track_sheet) if config.track_sheet else None
    dataset_dir = Path(config.dataset_dir) if config.dataset_dir else None
    groups = resolve_track_groups(
        ensemble.track_names,
        group_by=target.group_by,
        target=target.target,
        exclude_tracks=set(config.exclude_tracks),
        track_sheet=track_sheet,
        dataset_dir=dataset_dir,
    )
    logger.info(
        f"Target group {target.target!r}; {len(groups.other_group_masks)} off-target group(s)"
    )
    return groups


def _resolve_ism_hits(config: "DesignConfig") -> list[tuple[str, int, int]] | None:
    if config.ism_positions is None:
        return None
    from regulonado.genomics import read_intervals

    frame = read_intervals(config.ism_positions)[["chrom", "start", "end"]]
    return [
        (str(chrom), int(start), int(end))
        for chrom, start, end in frame.itertuples(index=False, name=None)
    ]


def _log_projected_passes(config: "DesignConfig", seeds: list["Seed"]) -> None:
    total_positions = sum(
        len(range(seed.editable.start, seed.editable.stop, config.ism_stride)) for seed in seeds
    )
    n_folds = len(config.checkpoint_dirs or [])
    logger.info(
        f"Projected ISM forward passes: ~{total_positions * 3 * config.rounds * n_folds:,} "
        f"({total_positions} position(s) x 3 alt bases x {config.rounds} round(s) x "
        f"{n_folds} fold(s))"
    )


def _energy_kwargs(config: "DesignConfig") -> dict[str, Any]:
    return dict(
        target_alpha=config.target_alpha,
        bending_factor=config.bending_factor,
        offtarget_reduction=config.offtarget_reduction,
        offtarget_temperature=config.offtarget_temperature,
        bin_reduction=config.bin_reduction,
        topk_bins=config.topk_bins,
        objective=config.objective,
        offtarget_boost_weight=config.offtarget_boost_weight,
        offtarget_boost_tolerance=config.offtarget_boost_tolerance,
        gain_transform=config.gain_transform,
        gain_pseudocount=config.gain_pseudocount,
    )


def _dispatch_search(
    config: "DesignConfig",
    method: str,
    energy_fn,
    seed: "Seed",
    context: np.ndarray,
    ism_hits: list[tuple[str, int, int]] | None,
    run_seed: int,
    candidate_index: int,
    on_round: Callable[[dict], None],
) -> "DesignState":
    from regulonado.design.search import AdaLeadConfig, adalead, ism_greedy

    rng = np.random.default_rng(np.random.SeedSequence([run_seed, candidate_index]))
    if method == "ism":
        positions = None
        if ism_hits is not None:
            positions = [
                hit_start + offset - seed.window.ctx_start
                for hit_chrom, hit_start, hit_end in ism_hits
                if hit_chrom == seed.chrom
                for offset in range(hit_end - hit_start)
            ]
        return ism_greedy(
            energy_fn,
            seed,
            context,
            rounds=config.rounds,
            top_k=config.top_k,
            positions=positions,
            stride=config.ism_stride,
            batch_size=config.batch_size,
            rng=rng,
            on_round=on_round,
        )
    adalead_config = AdaLeadConfig(
        rounds=config.rounds,
        population_size=config.population_size,
        mu=config.mu,
        recomb_rate=config.recomb_rate,
        model_queries_per_batch=config.model_queries_per_batch,
        threshold=config.threshold,
        rho=config.rho,
    )
    return adalead(energy_fn, seed, context, adalead_config, rng=rng, on_round=on_round)


def _make_on_round(
    seed_name: str, wandb_run, wandb_history: list[dict], ensemble, wandb_module
) -> Callable[[dict], None]:
    def _on_round(entry: dict) -> None:
        summary = ", ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in entry.items()
            if key not in ("round", "positions", "sequence")
        )
        logger.info(f"  [{seed_name}] round {entry['round']}: {summary}")
        if wandb_run is None:
            return
        wandb_history.append(dict(entry))
        table = _trajectory_table(wandb_module, wandb_history)
        scalars = {k: v for k, v in entry.items() if k not in ("positions", "sequence")}
        scalars.update(
            {
                f"track/{name}": entry[f"track_{index}"]
                for index, name in enumerate(ensemble.track_names)
                if f"track_{index}" in entry
            }
        )
        wandb_run.log({**scalars, "candidate_results": table}, step=entry["round"])

    return _on_round


def _start_wandb_run(
    wandb_module, config: "DesignConfig", seed: "Seed", method: str, target: "DesignTarget"
):
    return wandb_module.init(
        project=config.wandb_project,
        group=config.wandb_group or target.target,
        job_type=method,
        name=f"{seed.name}_{method}",
        reinit=True,
        config={
            "candidate": seed.name,
            "chrom": seed.chrom,
            "start": seed.cand_start,
            "end": seed.cand_end,
            "fold_label": seed.fold_label,
            "method": method,
            "objective": config.objective,
            "gain_transform": config.gain_transform,
            "gain_pseudocount": config.gain_pseudocount,
            "rounds": config.rounds,
            "target": target.target,
            "group_by": target.group_by,
            "wandb_group": config.wandb_group or target.target,
        },
    )


def _finish_wandb_run(
    wandb_module,
    wandb_run,
    state: "DesignState",
    baseline_energy: float,
    final_result,
    holdout_result,
) -> None:
    wandb_run.summary["final_energy"] = state.energy
    wandb_run.summary["baseline_energy"] = baseline_energy
    wandb_run.summary["final_specificity"] = float(final_result.specificity[0])
    wandb_run.summary["final_target_gain"] = float(final_result.target_gain[0])
    wandb_run.summary["final_offtarget_boost"] = float(final_result.offtarget_boost[0])
    if holdout_result is not None:
        wandb_run.summary["holdout_energy"] = float(holdout_result.energy[0])
        wandb_run.summary["holdout_target"] = float(holdout_result.target[0])
        wandb_run.summary["holdout_target_gain"] = float(holdout_result.target_gain[0])
        wandb_run.summary["holdout_offtarget_boost"] = float(holdout_result.offtarget_boost[0])
    # Keep the final complete snapshot under a stable key as well.
    table = _trajectory_table(wandb_module, state.history)
    wandb_run.log({"trajectory": table, "candidate_results": table}, step=len(state.history) - 1)
    wandb_run.finish()


def _score_one_candidate(
    config: "DesignConfig",
    candidate_index: int,
    total: int,
    seed: "Seed",
    method: str,
    ensemble: "FoldEnsemble",
    holdout_ensemble: "FoldEnsemble | None",
    groups,
    fasta,
    chrom_sizes: dict[str, int],
    ism_hits: list[tuple[str, int, int]] | None,
    run_seed: int,
    wandb_module,
    target: "DesignTarget",
):
    from regulonado.design.objective import SpecificityEnergy
    from regulonado.design.report import DesignRecord
    from regulonado.genomics import one_hot_context

    logger.info(
        f"[{candidate_index}/{total}] {seed.name} "
        f"({seed.chrom}:{seed.cand_start}-{seed.cand_end}), method={method}"
    )
    chrom_length = chrom_sizes.get(seed.chrom)
    if chrom_length is None:
        raise ValueError(f"Chromosome {seed.chrom!r} not present in {config.fasta}")
    context = one_hot_context(fasta, seed.window, ensemble.context_length, chrom_length)

    energy_fn = SpecificityEnergy(ensemble, groups, seed.bins, **_energy_kwargs(config))
    if config.objective == "selective-activation":
        energy_fn.set_reference(context[None])

    wandb_run = (
        _start_wandb_run(wandb_module, config, seed, method, target) if wandb_module else None
    )
    on_round = _make_on_round(seed.name, wandb_run, [], ensemble, wandb_module)
    state = _dispatch_search(
        config, method, energy_fn, seed, context, ism_hits, run_seed, candidate_index, on_round
    )
    final_result = energy_fn(state.context[None])

    holdout_result = None
    if holdout_ensemble is not None:
        holdout_energy_fn = SpecificityEnergy(
            holdout_ensemble, groups, seed.bins, **_energy_kwargs(config)
        )
        if config.objective == "selective-activation":
            holdout_energy_fn.set_reference(context[None])
        holdout_result = holdout_energy_fn(state.context[None])

    baseline_energy = state.history[0]["energy"]
    holdout_suffix = (
        f", holdout={float(holdout_result.energy[0]):.4f}" if holdout_result is not None else ""
    )
    logger.info(
        f"[{candidate_index}/{total}] {seed.name} done: energy {baseline_energy:.4f} -> "
        f"{state.energy:.4f}{holdout_suffix}"
    )
    if wandb_run is not None:
        _finish_wandb_run(
            wandb_module, wandb_run, state, baseline_energy, final_result, holdout_result
        )

    record = DesignRecord(
        seed=seed,
        method=method,
        original_context=context,
        state=state,
        result=final_result,
        holdout_result=holdout_result,
    )
    report_entry = {
        "name": seed.name,
        "chrom": seed.chrom,
        "start": seed.cand_start,
        "end": seed.cand_end,
        "window_start": seed.window.ctx_start,
        "window_end": seed.window.ctx_end,
        "fold_label": seed.fold_label,
    }
    return record, report_entry


def _process_candidates(
    config: "DesignConfig",
    seeds: list["Seed"],
    method: str,
    ensemble: "FoldEnsemble",
    holdout_ensemble: "FoldEnsemble | None",
    groups,
    fasta,
    chrom_sizes: dict[str, int],
    ism_hits: list[tuple[str, int, int]] | None,
    run_seed: int,
    wandb_module,
    target: "DesignTarget",
    out_dir: Path,
):
    from regulonado.design.report import write_designs

    records = []
    seed_report = []
    for candidate_index, seed in enumerate(seeds, start=1):
        record, report_entry = _score_one_candidate(
            config,
            candidate_index,
            len(seeds),
            seed,
            method,
            ensemble,
            holdout_ensemble,
            groups,
            fasta,
            chrom_sizes,
            ism_hits,
            run_seed,
            wandb_module,
            target,
        )
        records.append(record)
        seed_report.append(report_entry)
        # Persist completed candidates immediately so an interrupted shard can be
        # resumed/recovered without losing earlier oracle work.
        write_designs(
            out_dir,
            records,
            run_info={"status": "in_progress", "completed_candidates": len(records)},
        )
    return records, seed_report


def _build_run_info(
    config: "DesignConfig",
    intervals: Path,
    method: str,
    target: "DesignTarget",
    groups,
    run_seed: int,
    seed_report: list[dict],
    ensemble: "FoldEnsemble",
) -> dict[str, Any]:
    return {
        "candidates": str(config.candidates),
        "intervals": str(intervals),
        "fasta": str(config.fasta),
        "checkpoints": [str(c) for c in (config.checkpoint_dirs or [])],
        "holdout_checkpoint": str(config.holdout_checkpoint) if config.holdout_checkpoint else None,
        "target": target.target,
        "objective": config.objective,
        "group_by": target.group_by,
        "method": method,
        "rounds": config.rounds,
        "pad": config.pad,
        "on_missing": config.on_missing,
        "fold_mode": config.fold_mode,
        "track_groups": {"target": groups.target, "labels": groups.labels},
        "excluded_tracks": sorted(config.exclude_tracks),
        "seed_resolution": seed_report,
        "seed": run_seed,
        "topk_bins": config.topk_bins,
        "bin_reduction": config.bin_reduction,
        "target_alpha": config.target_alpha,
        "bending_factor": config.bending_factor,
        "offtarget_reduction": config.offtarget_reduction,
        "offtarget_temperature": config.offtarget_temperature,
        "offtarget_boost_weight": config.offtarget_boost_weight,
        "offtarget_boost_tolerance": config.offtarget_boost_tolerance,
        "gain_transform": config.gain_transform,
        "gain_pseudocount": config.gain_pseudocount,
        "batch_size": config.batch_size,
        "device": str(config.device or ("cuda" if torch.cuda.is_available() else "cpu")),
        "track_names": ensemble.track_names,
        "reproducibility": {
            "torch_deterministic": bool(torch.are_deterministic_algorithms_enabled())
        },
        "status": "complete",
    }


def run_design(config: "DesignConfig") -> DesignResult:
    """Mutate endogenous enhancer candidates to sharpen cell-type specificity.

    Optimises exactly the spans in ``config.candidates``; context for each comes from the
    dataset window (from ``config.intervals``) that contains it. ``config.targets`` must hold
    exactly one target for a standalone run (a multi-target params file describes a whole
    workflow ``design:`` section, not one invocation). Writes ``designs.{fa,tsv,bed}``,
    ``trajectory.tsv``, ``edits.tsv`` and ``run.json`` to ``config.out_dir``.
    """
    from regulonado.design.report import write_designs

    if len(config.targets) != 1:
        raise ValueError(
            f"run_design processes exactly one design target per call; got "
            f"{len(config.targets)} in config.targets. Call it once per target "
            f"(see docs/design.md)."
        )
    if not config.candidates:
        raise ValueError(
            "design.candidates is required for a standalone run; 'from_attribution' only "
            "resolves inside the packaged workflow, which passes the resolved BED as "
            "--candidates before calling this."
        )
    if not config.fasta:
        raise ValueError("design.fasta is required")
    if not config.out_dir:
        raise ValueError("design.out_dir is required")

    target = config.targets[0]
    method = target.method
    run_seed = _seed_everything(config.seed)
    intervals = _resolve_intervals(config)
    ensemble, holdout_ensemble = _load_folds(config)
    seeds = _resolve_seeds(config, ensemble, intervals)
    groups = _resolve_groups(config, target, ensemble)

    import pyfaidx

    fasta = pyfaidx.Fasta(str(config.fasta), as_raw=True, sequence_always_upper=False)
    chrom_sizes = {name: len(fasta[name]) for name in fasta.keys()}
    ism_hits = _resolve_ism_hits(config)
    if method == "ism":
        _log_projected_passes(config, seeds)

    wandb_module = None
    if config.wandb:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise ValueError(
                "design.wandb=true requires the 'wandb' package (part of regulonado[train])."
            ) from exc

    out_dir = Path(config.out_dir)
    records, seed_report = _process_candidates(
        config,
        seeds,
        method,
        ensemble,
        holdout_ensemble,
        groups,
        fasta,
        chrom_sizes,
        ism_hits,
        run_seed,
        wandb_module,
        target,
        out_dir,
    )
    run_info = _build_run_info(
        config, intervals, method, target, groups, run_seed, seed_report, ensemble
    )
    write_designs(out_dir, records, run_info=run_info)
    return DesignResult(
        out_dir=out_dir, target=target.target, method=method, n_candidates=len(records)
    )
