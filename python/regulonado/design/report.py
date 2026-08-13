"""Write design-run outputs: FASTA, TSVs, BED, and the resolved run manifest."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from regulonado.design.search import DesignState
from regulonado.design.sequence import Seed, decode

__all__ = ["DesignRecord", "write_designs", "hamming_distance"]

def hamming_distance(original: np.ndarray, final: np.ndarray, editable: slice) -> int:
    return int(np.any(original[:, editable] != final[:, editable], axis=0).sum())


@dataclass(slots=True)
class DesignRecord:
    """One search run (one candidate x one method), design-fold and held-out scores."""

    seed: Seed
    method: str
    original_context: np.ndarray  # (4, context_length), unedited — for the edits.tsv diff
    state: DesignState  # final design state (context/editable/energy/history)
    result: Any  # EnergyResult on the final context, scored on the design folds
    holdout_result: Any  # EnergyResult on the final context, scored on the held-out fold


def write_designs(out_dir: str | Path, records: list[DesignRecord], *, run_info: dict) -> None:
    """Write ``designs.{fa,tsv,bed}``, ``trajectory.tsv``, ``edits.tsv`` and ``run.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_fasta(out_dir / "designs.fa", records)
    _write_designs_tsv(out_dir / "designs.tsv", records)
    _write_designs_bed(out_dir / "designs.bed", records)
    _write_trajectory(out_dir / "trajectory.tsv", records)
    _write_edits(out_dir / "edits.tsv", records)
    _write_topk(out_dir / "topk.tsv", records)
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2, default=str))


def _write_fasta(path: Path, records: list[DesignRecord]) -> None:
    with path.open("w") as handle:
        for record in records:
            seed, state = record.seed, record.state
            insert = decode(state.context[:, state.editable])
            n_edits = hamming_distance(record.original_context, state.context, state.editable)
            header = (
                f">{seed.name}_{record.method} {seed.chrom}:{seed.cand_start}-{seed.cand_end} "
                f"energy={state.energy:.6g} n_edits={n_edits}"
            )
            handle.write(header + "\n")
            handle.write(insert + "\n")


def _group_columns(result: Any) -> dict[str, float]:
    if result is None:
        return {}
    return {
        f"group_{name}": float(value)
        for name, value in zip(result.group_names, result.per_group[0].tolist())
    }


def _write_designs_tsv(path: Path, records: list[DesignRecord]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        seed, state, result = record.seed, record.state, record.result
        n_edits = hamming_distance(record.original_context, state.context, state.editable)
        row: dict[str, Any] = {
            "name": seed.name,
            "chrom": seed.chrom,
            "start": seed.cand_start,
            "end": seed.cand_end,
            "window_chrom": seed.window.chrom,
            "window_start": seed.window.ctx_start,
            "window_end": seed.window.ctx_end,
            "fold_label": seed.fold_label or "",
            "method": record.method,
            "energy": float(state.energy),
            "objective": getattr(result, "objective", "specificity") if result is not None else "",
            "specificity": float(result.specificity[0]) if result is not None else "",
            "target_score": float(result.target[0]) if result is not None else "",
            "target_gain": float(result.target_gain[0]) if result is not None else "",
            "offtarget_boost": float(result.offtarget_boost[0]) if result is not None else "",
            "n_edits": n_edits,
            "per_fold_energy": (
                ",".join(f"{v:.6g}" for v in result.per_fold_energy[:, 0].tolist())
                if result is not None
                else ""
            ),
            "per_fold_specificity": (
                ",".join(f"{v:.6g}" for v in result.per_fold_specificity[:, 0].tolist())
                if result is not None
                else ""
            ),
        }
        row.update(_group_columns(result))
        if result is not None:
            for name, value in zip(result.group_names, result.per_group_gain[0].tolist()):
                row[f"group_gain_{name}"] = float(value)
        if result is not None and hasattr(result, "track_topk"):
            for i, value in enumerate(result.track_mean[0].tolist()):
                row[f"track_{i}_mean"] = float(value)
                row[f"track_{i}_max"] = float(result.track_max[0, i])
                row[f"track_{i}_topk"] = float(result.track_topk[0, i])
                row[f"track_{i}_topk_ratio"] = float(result.track_topk_ratio[0, i])
        holdout = record.holdout_result
        row["holdout_energy"] = float(holdout.energy[0]) if holdout is not None else ""
        row["holdout_specificity"] = (
            float(holdout.specificity[0]) if holdout is not None else ""
        )
        row["holdout_target_score"] = float(holdout.target[0]) if holdout is not None else ""
        row["holdout_target_gain"] = (
            float(holdout.target_gain[0]) if holdout is not None else ""
        )
        row["holdout_offtarget_boost"] = (
            float(holdout.offtarget_boost[0]) if holdout is not None else ""
        )
        for name, value in _group_columns(holdout).items():
            row[f"holdout_{name}"] = value
        if holdout is not None:
            for name, value in zip(holdout.group_names, holdout.per_group_gain[0].tolist()):
                row[f"holdout_group_gain_{name}"] = float(value)
        rows.append(row)

    _write_tsv(path, rows)


def _write_designs_bed(path: Path, records: list[DesignRecord]) -> None:
    seen: set[str] = set()
    with path.open("w") as handle:
        for record in records:
            seed = record.seed
            if seed.name in seen:
                continue
            seen.add(seed.name)
            handle.write(
                f"{seed.window.chrom}\t{seed.window.pred_start}\t{seed.window.pred_end}\t"
                f"{seed.name}\n"
            )


def _write_trajectory(path: Path, records: list[DesignRecord]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        for entry in record.state.history:
            row = {
                "name": record.seed.name,
                "method": record.method,
                "round": entry["round"],
                "energy": entry["energy"],
                "specificity": entry.get("specificity", ""),
                "target": entry.get("target", ""),
                "target_gain": entry.get("target_gain", ""),
                "offtarget_boost": entry.get("offtarget_boost", ""),
                "n_edits": entry.get("n_edits", ""),
                "sequence": entry.get("sequence", ""),
            }
            row.update(
                {
                    key: value
                    for key, value in entry.items()
                    if key.startswith("group_")
                }
            )
            rows.append(row)
    _write_tsv(path, rows)


def _write_edits(path: Path, records: list[DesignRecord]) -> None:
    rows: list[dict[str, Any]] = []
    bases = "ACGT"
    for record in records:
        seed, state = record.seed, record.state
        original = record.original_context
        final = state.context
        window = seed.window
        for position in range(state.editable.start, state.editable.stop):
            ref_col, alt_col = original[:, position], final[:, position]
            if np.array_equal(ref_col, alt_col):
                continue
            ref = bases[int(ref_col.argmax())] if ref_col.any() else "N"
            alt = bases[int(alt_col.argmax())] if alt_col.any() else "N"
            rows.append(
                {
                    "name": seed.name,
                    "method": record.method,
                    "chrom": window.chrom,
                    "position": window.ctx_start + position,
                    "ref": ref,
                    "alt": alt,
                }
            )
    _write_tsv(path, rows)


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_topk(path: Path, records: list[DesignRecord], k: int = 10) -> None:
    rows = []
    for record in records:
        result = record.result
        if result is None or not hasattr(result, "per_track"):
            continue
        values = (
            result.track_topk[0].tolist()
            if hasattr(result, "track_topk")
            else result.per_track[0].tolist()
        )
        means = (
            result.track_mean[0].tolist()
            if hasattr(result, "track_mean")
            else [None] * len(values)
        )
        maxima = (
            result.track_max[0].tolist()
            if hasattr(result, "track_max")
            else [None] * len(values)
        )
        ratios = (
            result.track_topk_ratio[0].tolist()
            if hasattr(result, "track_topk_ratio")
            else [None] * len(values)
        )
        for index, value in enumerate(values):
            rows.append(
                {
                    "name": record.seed.name,
                    "track_index": index,
                    "topk_mean": float(value),
                    "mean": float(means[index]),
                    "max": float(maxima[index]),
                    "topk_ratio": float(ratios[index]),
                    "k": k,
                }
            )
    _write_tsv(path, rows)
