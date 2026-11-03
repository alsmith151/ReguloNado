"""Prepare/run a small four-method comparison using the local AlphaGenome encoder.

Without --fasta, prepare coordinates/configs only. With --fasta, extract sequences,
label with the oracle, train a tiny MDLM, and run the four design methods.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_priority_regions(path):
    """Deduplicate loci while preserving all supplied specificity observations."""
    from regulonado.genomics import read_intervals

    intervals = read_intervals(path)
    raw = pd.read_csv(path, sep="\t", header=None, comment="#")
    if len(raw) != len(intervals) or raw.shape[1] < 8:
        raise ValueError("Priority BED must have eight columns with specificity in column 8")
    intervals["specificity"] = pd.to_numeric(raw[7], errors="raise")
    intervals["source_column7"] = raw[6].to_numpy()
    if not np.isfinite(intervals.specificity).all():
        raise ValueError("Priority specificity scores must be finite")
    records = []
    for (chrom, start, end), group in intervals.groupby(["chrom", "start", "end"], sort=False):
        records.append(
            dict(
                chrom=chrom,
                start=start,
                end=end,
                source_rows=len(group),
                HL60_specificity=float(group.specificity.mean()),
                specificity_observations=json.dumps(
                    group[["source_column7", "specificity"]].to_dict("records")
                ),
            )
        )
    return pd.DataFrame(records)


def prepare_regions(
    path,
    out,
    *,
    core_length=200,
    training_count=16,
    validation_count=4,
    candidate_count=2,
    seed=17,
    candidate_regions=None,
):
    from regulonado.genomics import read_intervals

    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        frame["dataset_row"] = np.arange(len(frame))
        frame["source_start"], frame["source_end"] = frame.start, frame.end
        if {"target_start", "target_end"} <= set(frame):
            frame = frame.assign(start=frame.target_start, end=frame.target_end)
    else:
        frame = read_intervals(path)
        frame["source_start"], frame["source_end"] = frame.start, frame.end
        center = (frame.start + frame.end) // 2
        frame = frame.assign(start=center - 500, end=center + 500)
    if candidate_regions is not None:
        priority = read_priority_regions(candidate_regions)
        keys = set(zip(priority.chrom, priority.start, priority.end))
        coordinate_columns = (
            ["orig_chrom", "orig_start", "orig_end"]
            if {"orig_chrom", "orig_start", "orig_end"} <= set(frame)
            else ["chrom", "source_start", "source_end"]
        )
        frame["is_priority"] = [
            tuple(row) in keys
            for row in frame[coordinate_columns].itertuples(index=False, name=None)
        ]
        score_by_region = dict(
            zip(zip(priority.chrom, priority.start, priority.end), priority.HL60_specificity)
        )
        frame["HL60_specificity"] = [
            score_by_region.get(tuple(row), np.nan)
            for row in frame[coordinate_columns].itertuples(index=False, name=None)
        ]
        matched = frame.loc[frame.is_priority].copy()
        if matched.empty:
            raise ValueError("No priority regions match original dataset intervals exactly")
        out.mkdir(parents=True, exist_ok=True)
        audit = matched[coordinate_columns + ["split"]].rename(
            columns=dict(zip(coordinate_columns, ["chrom", "start", "end"]))
        )
        audit = priority.merge(audit, on=["chrom", "start", "end"], how="left")
        audit.to_csv(out / "priority_regions.tsv", sep="\t", index=False)
    if "ccre_class" in frame:
        eligible = frame.ccre_class.isin(["dELS", "pELS"])
        if candidate_regions is not None:
            eligible |= frame.is_priority
        frame = frame.loc[eligible].copy()
    frame = frame.drop_duplicates(["chrom", "start", "end"]).copy()
    if (
        frame[["chrom", "start", "end"]].isna().any().any()
        or (frame.start < 0).any()
        or (frame.end <= frame.start).any()
    ):
        raise ValueError("Regions need valid chrom/start/end (0-based, half-open hg38 coordinates)")
    if not 1 <= core_length <= int((frame.end - frame.start).min()):
        raise ValueError("core_length must fit every target interval")
    if "split" in frame:
        frame["split"] = frame.split.replace({"val": "validation", "valid": "validation"})
        split_masks = {name: frame.split.eq(name) for name in ["train", "validation", "test"]}
    else:
        split_masks = {
            "train": ~frame.chrom.isin(["chr8", "chr9", "chr10"]),
            "validation": frame.chrom.isin(["chr8", "chr9"]),
            "test": frame.chrom.eq("chr10"),
        }
    if candidate_regions is not None:
        split_masks["test"] &= frame.is_priority
    chosen = []
    for split, count in zip(split_masks, [training_count, validation_count, candidate_count]):
        pool = frame.loc[split_masks[split]]
        if len(pool) < count:
            raise ValueError(f"Need {count} {split} regions, found {len(pool)}")
        if candidate_regions is not None and split != "test":
            preferred = pool.loc[pool.is_priority]
            background = pool.loc[~pool.is_priority]
            n_preferred = min(len(preferred), max(count // 2, count - len(background)))
            part = pd.concat(
                [
                    preferred.sample(n=n_preferred, random_state=seed),
                    background.sample(n=count - n_preferred, random_state=seed),
                ]
            ).copy()
        else:
            part = pool.sample(n=count, random_state=seed).copy()
        part["split"] = split
        chosen.append(part)
    selected = pd.concat(chosen, ignore_index=True)
    if selected.groupby("chrom").split.nunique().gt(1).any():
        raise ValueError("Chromosomes must not cross split boundaries")
    center = (selected.start + selected.end) // 2
    selected["core_start"] = center - core_length // 2
    selected["core_end"] = selected.core_start + core_length
    selected["name"] = [f"encoder_smoke_{i}" for i in range(len(selected))]
    if (selected.start < 0).any():
        raise ValueError("A selected region is too close to the start of a chromosome")
    if selected.duplicated(["chrom", "start", "end"]).any():
        raise ValueError(
            "Distinct input intervals map to identical anchors; deduplicate by midpoint"
        )
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "selected_regions.tsv", sep="\t", index=False)
    selected[["chrom", "start", "end", "split"]].to_csv(
        out / "anchors.bed", sep="\t", header=False, index=False
    )
    selected.loc[selected.split.eq("test"), ["chrom", "core_start", "core_end", "name"]].to_csv(
        out / "candidates.bed", sep="\t", header=False, index=False
    )
    return selected


def write_configs(
    out,
    checkpoint,
    *,
    fasta=None,
    activities=None,
    core_length=200,
    device="auto",
    validation_chromosomes=None,
):
    from regulonado.config.models import DesignConfig
    from regulonado.design.diffusion import DiffusionConfig, device_for

    device = str(device_for(device))
    training = DiffusionConfig(
        activity_columns=list(activities or {"HL-60": 0.0, "K-562": 0.0}),
        max_length=core_length,
        width=32,
        heads=4,
        layers=2,
        epochs=3,
        batch_size=4,
        device=device,
        validation_chromosomes=validation_chromosomes or ["chr8", "chr9"],
        seed=17,
    )
    (out / "diffusion-training.yaml").write_text(yaml.safe_dump(training.model_dump()))
    for method in ["ism", "adalead", "gradient", "diffusion"]:
        settings = dict(
            candidates=str(out / "candidates.bed"),
            intervals=str(out / "anchors.bed"),
            checkpoint_dirs=[str(checkpoint)],
            model_kind="region_counts",
            energy="contrast",
            objective="selective-activation",
            offtarget_boost_weight=1.0,
            offtarget_boost_tolerance=0.0,
            bin_reduction="mean",
            gain_transform="raw",
            device=device,
            batch_size=1,
            fold_mode="resident",
            rounds=1,
            seed=17,
            out_dir=str(out / "designs" / method),
            targets=[dict(name=f"hl60_{method}", target="HL-60", method=method)],
        )
        if fasta:
            settings["fasta"] = str(fasta)
        if method == "ism":
            settings.update(ism_stride=50, top_k=1)
        elif method == "adalead":
            settings.update(population_size=4, model_queries_per_batch=12)
        elif method == "gradient":
            settings.update(rounds=8, gradient_learning_rate=0.5)
        else:
            settings.update(
                diffusion_mode="de_novo",
                diffusion_checkpoint=str(out / "mdlm" / "best.pt"),
                diffusion_activities=activities or {},
                diffusion_samples=4,
                diffusion_steps=8,
            )
        config = DesignConfig.model_validate(settings)
        (out / f"{method}.yaml").write_text(
            yaml.safe_dump(config.model_dump(exclude_unset=True), sort_keys=False)
        )
    profiles_path = out / "specificity_profiles.json"
    if profiles_path.exists():
        base = yaml.safe_load((out / "diffusion.yaml").read_text())
        for name, profile in json.loads(profiles_path.read_text()).items():
            config = dict(base)
            config["diffusion_activities"] = profile["activities"]
            config["out_dir"] = str(out / "designs" / f"diffusion_{name}")
            config["targets"] = [
                dict(name=f"hl60_diffusion_{name}", target="HL-60", method="diffusion")
            ]
            validated = DesignConfig.model_validate(config)
            (out / f"diffusion-{name}.yaml").write_text(
                yaml.safe_dump(validated.model_dump(exclude_unset=True), sort_keys=False)
            )
    return training


def label_regions(selected, fasta_path, checkpoint, out, device, regions_path):
    import pyfaidx
    from regulonado.design.region_predictor import RegionSequencePredictor
    from regulonado.genomics import decode, one_hot_context, window_for_interval

    oracle = RegionSequencePredictor(checkpoint, device=device, batch_size=1)
    rows = []
    with pyfaidx.Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True) as fasta:
        for row in selected.itertuples():
            if row.chrom not in fasta:
                raise ValueError(f"Missing chromosome {row.chrom} in FASTA")
            if row.end > len(fasta[row.chrom]):
                raise ValueError(f"Region {row.name} exceeds chromosome length")
            window = window_for_interval(
                row.chrom,
                row.start,
                row.end,
                context_length=oracle.context_length,
                n_pred_bins=1,
                bin_size=oracle.bin_size,
                crop_bp=oracle.crop_bp,
                snap_bp=oracle.snap_bp,
            )
            core = slice(row.core_start - window.ctx_start, row.core_end - window.ctx_start)
            if not window.pred_start <= row.core_start < row.core_end <= window.pred_end:
                raise ValueError(f"Core {row.name} falls outside oracle's scored span")
            context = one_hot_context(fasta, window, oracle.context_length, len(fasta[row.chrom]))
            sequence = decode(context[:, core])
            if set(sequence) - set("ACGT"):
                raise ValueError(f"Core {row.name} contains ambiguous bases; choose another region")
            scores = oracle(context[None])[0, :, 0].float().cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError(f"Nonfinite oracle scores for {row.name}")
            rows.append(
                dict(
                    chrom=row.chrom,
                    name=row.name,
                    sequence=sequence,
                    split=row.split,
                    **dict(zip(oracle.group_names, scores.tolist())),
                )
            )
            print(f"Scored {row.name} ({row.split})", flush=True)
    table = pd.DataFrame(rows)
    if "HL60_specificity" in selected:
        table["HL60_specificity"] = selected.HL60_specificity.to_numpy()
    table.to_csv(out / "oracle_labels.tsv", sep="\t", index=False)
    training_labels = table.copy()
    label_source = "oracle predictions"
    dataset_dir = Path(regions_path).parent
    if "dataset_row" in selected and (dataset_dir / "counts.parquet").is_file():
        import polars as pl
        from regulonado.counts.dataset import group_count_rates

        config = json.loads((Path(checkpoint) / "config.json").read_text())
        counts = (
            pl.scan_parquet(dataset_dir / "counts.parquet")
            .with_row_index("dataset_row")
            .filter(pl.col("dataset_row").is_in(selected.dataset_row.tolist()))
            .collect(engine="streaming")
            .to_pandas()
            .set_index("dataset_row")
            .loc[selected.dataset_row]
        )
        values = counts[config["track_names"]].to_numpy(dtype=float)
        rates = group_count_rates(
            values,
            np.array(config["track_groups"]),
            np.array(config["log_size_factors"]),
            config["n_groups"],
        )
        training_labels[oracle.group_names] = np.log1p(rates)
        training_labels.to_csv(out / "measured_labels.tsv", sep="\t", index=False)
        label_source = (
            "measured counts pooled by group using checkpoint log-size factors, then log1p"
        )
    training_labels.loc[training_labels.split.ne("test")].to_csv(
        out / "diffusion-training.tsv", sep="\t", index=False
    )
    # Request an observed training profile, rather than arbitrary extrapolated numbers.
    train = training_labels.loc[training_labels.split.eq("train")]
    others = [group for group in oracle.group_names if group != "HL-60"]
    known = (
        train.loc[train.HL60_specificity.notna()].sort_values("HL60_specificity")
        if "HL60_specificity" in train
        else train.iloc[:0]
    )
    condition_columns = list(oracle.group_names)
    if not known.empty:
        condition_columns.append("HL60_specificity")
        best = known.HL60_specificity.idxmax()
        profiles = {}
        for fraction in [0.25, 0.5, 0.75]:
            row = known.iloc[round(fraction * (len(known) - 1))]
            profiles[f"q{int(fraction * 100)}"] = dict(
                source=row["name"],
                activities={column: float(row[column]) for column in condition_columns},
            )
        (out / "specificity_profiles.json").write_text(json.dumps(profiles, indent=2))
    else:
        best = (train["HL-60"] - train[others].mean(axis=1)).idxmax()
    activities = {column: float(training_labels.loc[best, column]) for column in condition_columns}
    metadata = dict(
        checkpoint=str(checkpoint),
        reference=str(fasta_path),
        target="HL-60",
        score_units="softplus(eta) = log1p(rate)",
        groups=oracle.group_names,
        device=device,
        specificity_aggregation="Arithmetic mean of per-assay scores per original region",
        condition_source=table.loc[best, "name"],
        conditioning_profile=activities,
        input_length=oracle.context_length,
        scored_length=oracle.bin_size,
        context_note=(
            "Original target windows preserved for parquet; "
            "BED input uses new centred 1000bp anchors"
        ),
        training_label_source=label_source,
        split_note="Splits hold out DNA from the denoiser, not necessarily from this oracle",
    )
    (out / "oracle.json").write_text(json.dumps(metadata, indent=2))
    return activities


def summarize(out, elapsed=None):
    """Collect scores and select a small panel without rerunning any model."""
    summary = []
    for method in ["ism", "adalead", "gradient", "diffusion"]:
        designs = pd.read_csv(out / "designs" / method / "designs.tsv", sep="\t")
        trajectory = pd.read_csv(out / "designs" / method / "trajectory.tsv", sep="\t")
        for design in designs.to_dict("records"):
            baseline = trajectory.loc[
                (trajectory.name == design["name"]) & trajectory["round"].eq(0)
            ].iloc[0]
            summary.append(
                dict(
                    method=method,
                    name=design["name"],
                    baseline_energy=float(baseline.energy),
                    final_energy=design["energy"],
                    energy_improvement=float(baseline.energy) - design["energy"],
                    baseline_target=float(baseline.target),
                    final_target=design["target_score"],
                    n_edits=design["n_edits"],
                    method_seconds=(elapsed or {}).get(method, float("nan")),
                )
            )
    pd.DataFrame(summary).to_csv(out / "summary.tsv", sep="\t", index=False)
    from regulonado.cli.generate import select

    select(
        tables=[
            out / "designs" / method / "trajectory.tsv"
            for method in ["ism", "adalead", "gradient", "diffusion"]
        ],
        out=out / "panel",
        count=8,
        min_distance=10,
        score="energy",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        type=Path,
        default=ROOT / "models_from_ccb/region_counts/dataset/regions.parquet",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models_from_ccb/alphagenome-encoder"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "local/alphagenome_encoder_hl60_specificity_smoke"
    )
    parser.add_argument(
        "--candidate-regions",
        type=Path,
        default=ROOT / "models_from_ccb/regions_more_specific_to_HL60.bed",
        help="Curated HL-60 BED; use its test-split loci as design candidates",
    )
    parser.add_argument("--fasta", type=Path, help="hg38 FASTA; omit to prepare inputs only")
    parser.add_argument("--device", default="auto", help="auto prefers CUDA/MPS, then CPU")
    parser.add_argument("--core-length", type=int, default=200)
    parser.add_argument("--train-regions", type=int, default=16)
    parser.add_argument("--validation-regions", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument(
        "--prepare-only", action="store_true", help="Do not label, train, or design"
    )
    args = parser.parse_args()
    from regulonado.design.diffusion import device_for

    args.device = str(device_for(args.device))
    print(f"Using device: {args.device}", flush=True)
    out, checkpoint = args.out.resolve(), args.checkpoint.resolve()
    fasta = args.fasta.resolve() if args.fasta else None
    if fasta and not fasta.is_file():
        parser.error(f"FASTA does not exist: {fasta}")
    selected = prepare_regions(
        args.regions,
        out,
        core_length=args.core_length,
        training_count=args.train_regions,
        validation_count=args.validation_regions,
        candidate_count=args.candidates,
        candidate_regions=args.candidate_regions,
    )
    write_configs(
        out,
        checkpoint,
        fasta=fasta,
        core_length=args.core_length,
        device=args.device,
        validation_chromosomes=sorted(
            selected.loc[selected.split.eq("validation"), "chrom"].unique()
        ),
    )
    print(f"Prepared {len(selected)} regions and four method configs in {out}", flush=True)
    if not fasta or args.prepare_only:
        print("Preparation only; supply --fasta /path/to/hg38.fa to run the test.")
        return
    activities = label_regions(selected, fasta, checkpoint, out, args.device, args.regions)
    training = write_configs(
        out,
        checkpoint,
        fasta=fasta,
        activities=activities,
        validation_chromosomes=sorted(
            selected.loc[selected.split.eq("validation"), "chrom"].unique()
        ),
        core_length=args.core_length,
        device=args.device,
    )
    from regulonado.config.models import DesignConfig
    from regulonado.design.diffusion import train_diffusion
    from regulonado.design.run import run_design

    train_diffusion(out / "diffusion-training.tsv", out / "mdlm", training)
    elapsed = {}
    for method in ["ism", "adalead", "gradient", "diffusion"]:
        print(f"Running {method}", flush=True)
        started = time.perf_counter()
        payload = yaml.safe_load((out / f"{method}.yaml").read_text())
        if method == "diffusion":
            payload.setdefault("diffusion_mode", "de_novo")
        run_design(DesignConfig.model_validate(payload))
        elapsed[method] = time.perf_counter() - started
    summarize(out, elapsed)
    print(f"Smoke test complete: {out / 'designs'}")


if __name__ == "__main__":
    main()
