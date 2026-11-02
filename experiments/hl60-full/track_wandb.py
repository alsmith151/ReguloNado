"""Metrics-only W&B tracking. Never upload sequences, coordinates, tables, or files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import yaml


def safe_config(out):
    """Explicit numeric/public-label whitelist; no paths, regions, or sequence data."""
    manifest = json.loads((out / "manifest.json").read_text())
    training = yaml.safe_load((out / "training.yaml").read_text())
    fields = [
        "architecture",
        "encoder_compute_dtype",
        "stratified_timesteps",
        "width",
        "layers",
        "heads",
        "batch_size",
        "epochs",
        "learning_rate",
        "condition_dropout",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "early_stopping_min_epochs",
        "diagnostic_examples",
        "seed",
    ]
    return dict(
        target="HL-60",
        device=manifest["device"],
        core_length=manifest["core_length"],
        split_counts=manifest["split_counts"],
        training={k: training[k] for k in fields if k in training},
        condition_dimensions=len(training["activity_columns"]),
        specificity_sampling_weight=training.get("condition_sampling_weights", {}).get(
            "HL60_specificity", 1
        ),
        encoder_frozen=training.get("architecture") == "alphagenome_encoder",
        trainable_parameters=manifest.get("trainable_parameters"),
        frozen_parameters=manifest.get("frozen_parameters"),
        methods=manifest.get("methods", ["ism", "adalead", "gradient", "diffusion"]),
        payload="aggregate metrics only",
    )


def aggregate_metrics(frame, method):
    """All requested design measurements, reduced across loci; never export identifiers."""
    frame = frame.copy()
    gains = [
        column
        for column in frame
        if column.startswith("group_gain_") and column != "group_gain_HL-60"
    ]
    offscores = [
        column
        for column in frame
        if column.startswith("group_")
        and not column.startswith("group_gain_")
        and column != "group_HL-60"
    ]
    if gains:
        frame["worst_offtarget_gain"] = frame[gains].max(axis=1)
        frame["mean_offtarget_gain"] = frame[gains].mean(axis=1)
        frame["boosted_offtarget_groups"] = frame[gains].gt(0).sum(axis=1)
    if offscores and "group_HL-60" in frame:
        frame["target_minus_mean_offtarget"] = frame["group_HL-60"] - frame[offscores].mean(axis=1)
        frame["target_minus_max_offtarget"] = frame["group_HL-60"] - frame[offscores].max(axis=1)
    basic = {
        "energy",
        "specificity",
        "target_score",
        "target_gain",
        "offtarget_boost",
        "n_edits",
        "edit_penalty",
        "score_genomic",
        "score_neutral",
        "flank_delta",
        "worst_offtarget_gain",
        "mean_offtarget_gain",
        "boosted_offtarget_groups",
        "target_minus_mean_offtarget",
        "target_minus_max_offtarget",
    }
    metrics = {f"design/{method}/completed": len(frame)}
    for column in frame:
        is_group = column.startswith("group_")
        if column not in basic and not is_group:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        stats = {"mean": float(values.mean()), "median": float(values.median())}
        if not is_group:
            stats.update(
                min=float(values.min()), max=float(values.max()), std=float(values.std(ddof=0))
            )
        for stat, value in stats.items():
            metrics[f"design/{method}/{column}_{stat}"] = value
    if "target_gain" in frame:
        metrics[f"design/{method}/positive_target_gain_fraction"] = float(
            frame.target_gain.gt(0).mean()
        )
    # Preserve old chart names while adding the complete metric set.
    for column, alias in [
        ("target_gain", "median_target_gain"),
        ("offtarget_boost", "median_offtarget_boost"),
        ("n_edits", "median_edits"),
    ]:
        if column in frame:
            metrics[f"design/{method}/{alias}"] = float(frame[column].median())
    return metrics


def sync(run, out, seen, logged_epochs):
    history = out / "mdlm/history.json"
    if history.exists():
        for row in json.loads(history.read_text()):
            if row["epoch"] not in logged_epochs:
                metrics = {
                    "training/epoch": int(row["epoch"]),
                    "training/loss": float(row["train_loss"]),
                    "training/validation_loss": float(row["validation_loss"]),
                    "training/mdlm_objective_nats_per_base": float(row["train_loss"]),
                    "validation/mdlm_objective_nats_per_base": float(row["validation_loss"]),
                }
                for key, value in row.items():
                    for prefix, namespace in [
                        ("train_masked_", "training"),
                        ("validation_masked_", "validation"),
                    ]:
                        if key.startswith(prefix):
                            metrics[f"{namespace}/masked_token_{key.removeprefix(prefix)}"] = float(
                                value
                            )
                run.log(metrics)
                logged_epochs.add(row["epoch"])
    if history.exists() and not (out / "mdlm/training_summary.json").exists():
        rows = json.loads(history.read_text())
        if rows:
            best = min(rows, key=lambda row: row["validation_loss"])
            run.summary.update(
                {
                    "training/status": "running",
                    "training/early_stopped": False,
                    "training/best_epoch": best["epoch"],
                    "training/best_validation_loss": best["validation_loss"],
                    "training/epochs_completed": len(rows),
                }
            )
    path = out / "mdlm/training_summary.json"
    if path.exists():
        summary = json.loads(path.read_text())
        run.summary["training/status"] = "completed"
        for key in ["best_epoch", "best_validation_loss", "epochs_completed", "early_stopped"]:
            run.summary[f"training/{key}"] = summary[key]
    for method in ["ism", "adalead", "gradient", "diffusion"]:
        path = out / "designs" / method / "designs.tsv"
        if not path.exists() or seen.get(str(path)) == path.stat().st_mtime_ns:
            continue
        try:
            frame = pd.read_csv(path, sep="\t")
            if frame.empty:
                continue
            # Never pass the dataframe to W&B; send aggregate scalars only.
            metrics = aggregate_metrics(frame, method)
            run.log(metrics)
            run.summary.update(metrics)
            seen[str(path)] = path.stat().st_mtime_ns
            print(f"Synced aggregate {method} metrics: {len(frame)} designs", flush=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, AttributeError):
            continue
    for filename in ["method_summary.tsv", "control_summary.tsv", "graded_summary.tsv"]:
        path = out / filename
        if path.exists() and seen.get(str(path)) != path.stat().st_mtime_ns:
            frame = pd.read_csv(path, sep="\t", index_col=0)
            for label, row in frame.iterrows():
                for metric, value in row.items():
                    if pd.notna(value):
                        run.summary[f"results/{filename.removesuffix('.tsv')}/{label}/{metric}"] = (
                            float(value)
                        )
            seen[str(path)] = path.stat().st_mtime_ns
    performance = out / "performance.json"
    if performance.exists():
        for method, values in json.loads(performance.read_text()).items():
            for metric, value in values.items():
                if isinstance(value, (int, float)):
                    run.summary[f"performance/{method}/{metric}"] = value
    for method in ["ism", "adalead", "gradient", "diffusion"]:
        path = out / "designs" / method / "complete.json"
        if path.exists() and seen.get(str(path)) != path.stat().st_mtime_ns:
            numeric = {
                f"performance/{method}/{key}": float(value)
                for key, value in json.loads(path.read_text()).items()
                if isinstance(value, (int, float))
            }
            run.log(numeric)
            run.summary.update(numeric)
            seen[str(path)] = path.stat().st_mtime_ns
    path = out / "conditioning_ablation.json"
    if path.exists() and seen.get(str(path)) != path.stat().st_mtime_ns:
        for row in json.loads(path.read_text()):
            run.summary[f"conditioning/{row['conditioning']}/validation_loss"] = float(
                row["validation_loss"]
            )
        seen[str(path)] = path.stat().st_mtime_ns
    path = out / "paired_diagnostics.json"
    if path.exists() and seen.get(str(path)) != path.stat().st_mtime_ns:

        def numeric_leaves(value, prefix):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from numeric_leaves(child, f"{prefix}/{key}")
            elif isinstance(value, (int, float)) and np.isfinite(value):
                yield prefix, float(value)

        run.summary.update(dict(numeric_leaves(json.loads(path.read_text()), "diagnostics")))
        seen[str(path)] = path.stat().st_mtime_ns
    return logged_epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("local/hl60_full_2026-11-02"))
    parser.add_argument("--project", default="regulonado-design")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default="hl60-full-2026-11-02")
    parser.add_argument("--group", default="hl60-full")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--replay-training", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    state_path = out / "wandb_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"id": wandb.util.generate_id()}
    )
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=state["id"],
        resume="allow",
        name=args.name,
        group=args.group,
        job_type="method-comparison",
        dir=str(out),
        config=safe_config(out),
        allow_val_change=True,
        notes=(
            "MDLM objective: diffusion-weighted denoising cross-entropy in nats per valid "
            "DNA base, "
            "averaged over variable masking rates. Training oversamples specificity-labelled "
            "regions; validation averages regions uniformly. Those objective curves are not "
            "a like-for-like generalisation gap. Fixed-mask token NLL is unweighted "
            "cross-entropy in nats per masked base at 15%, 50%, or 90% masks. Accuracy is "
            "the fraction of masked bases reconstructed correctly. Neither loss is an "
            "oracle-activity loss. See the explicit-unit dashboard panels."
        ),
        save_code=False,
        settings=wandb.Settings(
            disable_git=True,
            disable_code=True,
            disable_job_creation=True,
            x_disable_meta=True,
            x_disable_stats=True,
            x_disable_machine_info=True,
            console="off",
        ),
    )
    run.define_metric("training/epoch")
    run.define_metric("training/*", step_metric="training/epoch")
    run.define_metric("validation/*", step_metric="training/epoch")
    for method in ["ism", "adalead", "gradient", "diffusion"]:
        run.define_metric(f"design/{method}/completed")
        run.define_metric(f"design/{method}/*", step_metric=f"design/{method}/completed")
    state.update(
        url=run.url,
        entity=run.entity,
        project=run.project,
        status="running",
        payload="metrics_only",
    )
    state_path.write_text(json.dumps(state, indent=2))
    print(f"WANDB_RUN_URL={run.url}", flush=True)
    run.summary["status"] = "running"
    seen = {}
    last_epoch = int(run.summary.get("training/epoch", 0))
    logged_epochs = set() if args.replay_training else set(range(1, last_epoch + 1))
    while True:
        try:
            sync(run, out, seen, logged_epochs)
        except (json.JSONDecodeError, FileNotFoundError):
            time.sleep(1)
            continue
        complete = (out / "paired_diagnostics.json").exists() and (out / "results.png").exists()
        if complete or not args.follow:
            break
        time.sleep(10)
    if complete:
        run.summary["status"] = "complete"
        state["status"] = "complete"
        state_path.write_text(json.dumps(state, indent=2))
    run.finish()


if __name__ == "__main__":
    main()
