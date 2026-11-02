"""Reproducible local HL-60 enhancer design experiment; model computation on MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import runpy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
HELPERS = runpy.run_path(str(ROOT / "experiments/alphagenome-encoder-smoke/run.py"))
METHODS = ["ism", "adalead", "gradient", "diffusion"]


def canonical(sequence):
    return min(sequence, sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1])


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def prepare(args, out):
    import polars as pl
    import pyfaidx
    from regulonado.counts.dataset import group_count_rates
    from regulonado.design.region_predictor import RegionSequencePredictor
    from regulonado.genomics import one_hot_context, window_for_interval

    selected = HELPERS["prepare_regions"](
        args.regions,
        out,
        core_length=200,
        training_count=8192,
        validation_count=1024,
        candidate_count=20,
        candidate_regions=args.candidates,
    )
    # Core selection is part of preparation, not a design method's optimisation.
    oracle = RegionSequencePredictor(args.checkpoint, device=args.device, batch_size=8)
    core_records = []
    with pyfaidx.Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True) as fasta:
        for index, row in selected.loc[selected.split.eq("test")].iterrows():
            window = window_for_interval(
                row.chrom,
                row.start,
                row.end,
                context_length=4096,
                n_pred_bins=1,
                bin_size=oracle.bin_size,
                crop_bp=oracle.crop_bp,
                snap_bp=128,
            )
            context = one_hot_context(fasta, window, 4096, len(fasta[row.chrom]))
            gradient, score = oracle.gradient(
                context, [oracle.group_names.index("HL-60")], slice(0, 1)
            )
            attribution = np.maximum((gradient * context).sum(axis=0), 0)
            lo = max(window.pred_start, int(row.orig_start)) - window.ctx_start
            hi = min(window.pred_end, int(row.orig_end)) - window.ctx_start
            if hi - lo < 200:
                raise ValueError(f"No 200 bp editable core fits {row['name']}")
            starts = np.arange(lo, hi - 200 + 1)
            rolling = np.convolve(attribution, np.ones(200), mode="valid")
            valid_bases = np.convolve(context.any(axis=0).astype(int), np.ones(200), mode="valid")
            starts = starts[valid_bases[starts] == 200]
            if not len(starts):
                raise ValueError(f"No unambiguous core for {row['name']}")
            start = int(starts[np.argmax(rolling[starts])]) + window.ctx_start
            selected.loc[index, ["core_start", "core_end"]] = [start, start + 200]
            core_records.append(
                dict(
                    name=row["name"],
                    original_start=int(row.core_start),
                    core_start=start,
                    core_end=start + 200,
                    attribution_sum=float(rolling[start - window.ctx_start]),
                    baseline_centered_score=float(score),
                )
            )
        sequences = [
            str(fasta[row.chrom][int(row.core_start) : int(row.core_end)]).upper()
            for row in selected.itertuples()
        ]
    selected["sequence"] = sequences
    selected["canonical"] = selected.sequence.map(canonical)
    test_keys = set(selected.loc[selected.split.eq("test"), "canonical"])
    valid = selected.sequence.str.fullmatch("[ACGT]{200}")
    valid &= selected.split.eq("test") | ~selected.canonical.isin(test_keys)
    # Avoid exact/RC duplicates, including cross-split copies, without relabelling DNA.
    cross = selected.loc[valid].groupby("canonical").split.nunique()
    valid &= ~selected.canonical.isin(set(cross[cross > 1].index))
    before = len(selected)
    selected = selected.loc[valid].drop_duplicates(["split", "canonical"]).reset_index(drop=True)
    if selected.loc[selected.split.eq("test")].shape[0] != 20:
        raise ValueError("Test-core filtering changed the preselected 20 loci")
    logging.info(
        "Prepared %s; removed %d ambiguous/duplicate/leaking sequences",
        selected.split.value_counts().to_dict(),
        before - len(selected),
    )
    selected.to_csv(out / "selected_regions.tsv", sep="\t", index=False)
    selected[["chrom", "start", "end", "split"]].to_csv(
        out / "anchors.bed", sep="\t", index=False, header=False
    )
    selected.loc[selected.split.eq("test"), ["chrom", "core_start", "core_end", "name"]].to_csv(
        out / "candidates.bed", sep="\t", index=False, header=False
    )
    pd.DataFrame(core_records).to_csv(out / "core_attribution.tsv", sep="\t", index=False)
    config = json.loads((args.checkpoint / "config.json").read_text())
    counts = (
        pl.scan_parquet(args.regions.parent / "counts.parquet")
        .with_row_index("dataset_row")
        .filter(pl.col("dataset_row").is_in(selected.dataset_row.tolist()))
        .collect(engine="streaming")
        .to_pandas()
        .set_index("dataset_row")
        .loc[selected.dataset_row]
    )
    rates = group_count_rates(
        counts[config["track_names"]].to_numpy(),
        np.array(config["track_groups"]),
        np.array(config["log_size_factors"]),
        config["n_groups"],
    )
    labels = selected[["chrom", "name", "sequence", "split", "HL60_specificity"]].copy()
    labels[config["group_names"]] = np.log1p(rates)
    labels.to_csv(out / "all_labels.tsv", sep="\t", index=False)
    training = labels.loc[labels.split.ne("test")]
    training.to_csv(out / "training.tsv", sep="\t", index=False)
    known = training.loc[
        training.split.eq("train") & training.HL60_specificity.notna()
    ].sort_values("HL60_specificity")
    median = known.iloc[len(known) // 2]
    profiles = {}
    others = [name for name in config["group_names"] if name != "HL-60"]
    measured_contrast = known["HL-60"] - known[others].mean(axis=1)
    slope, intercept = np.polyfit(
        known.HL60_specificity.to_numpy(), measured_contrast.to_numpy(), 1
    )
    save_json(
        out / "specificity_calibration.json",
        dict(
            slope=float(slope),
            intercept=float(intercept),
            fit="measured target-minus-mean-offtarget = slope * supplied specificity + intercept",
            training_loci=len(known),
        ),
    )
    columns = [*config["group_names"], "HL60_specificity"]
    for name, quantile in [("q25", 0.25), ("q50", 0.5), ("q75", 0.75)]:
        values = {column: float(median[column]) for column in columns}
        values["HL60_specificity"] = float(known.HL60_specificity.quantile(quantile))
        values["HL-60"] = float(
            median[others].mean() + slope * values["HL60_specificity"] + intercept
        )
        profiles[name] = values
    save_json(out / "conditioning_profiles.json", profiles)
    with args.checkpoint.joinpath("model.safetensors").open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    save_json(
        out / "manifest.json",
        dict(
            seed=17,
            device=args.device,
            reference=str(args.fasta.resolve()),
            regions=str(args.regions.resolve()),
            priority_bed=str(args.candidates.resolve()),
            checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=checkpoint_hash,
            split_counts=selected.split.value_counts().to_dict(),
            labelled_specificity=selected.groupby("split").HL60_specificity.count().to_dict(),
            dropped_sequences=before - len(selected),
            core_length=200,
            evaluation_limit=(
                "One oracle; these are model predictions, not independent functional validation"
            ),
        ),
    )
    logging.info("Prepared measured numeric labels and paired conditioning profiles")


def configs(args, out):
    from regulonado.config.models import DesignConfig
    from regulonado.design.diffusion import DiffusionConfig

    oracle_config = json.loads((args.checkpoint / "config.json").read_text())
    training = DiffusionConfig(
        activity_columns=[*oracle_config["group_names"], "HL60_specificity"],
        max_length=200,
        width=128,
        layers=4,
        heads=4,
        batch_size=64,
        epochs=200,
        learning_rate=0.0003,
        condition_dropout=0.15,
        condition_sampling_weights={"HL60_specificity": 16.0},
        early_stopping_patience=40,
        early_stopping_min_delta=0.0001,
        early_stopping_min_epochs=40,
        diagnostic_examples=128,
        validation_chromosomes=["chr8", "chr10"],
        device=args.device,
        seed=17,
    )
    (out / "training.yaml").write_text(yaml.safe_dump(training.model_dump()))
    profiles = json.loads((out / "conditioning_profiles.json").read_text())
    for method in METHODS:
        settings = dict(
            candidates=str(out / "candidates.bed"),
            intervals=str(out / "anchors.bed"),
            checkpoint_dirs=[str(args.checkpoint.resolve())],
            fasta=str(args.fasta.resolve()),
            model_kind="region_counts",
            energy="contrast",
            objective="selective-activation",
            offtarget_boost_weight=2.0,
            offtarget_temperature=0.25,
            offtarget_boost_tolerance=0.0,
            gain_transform="raw",
            bin_reduction="mean",
            device=args.device,
            batch_size=8,
            fold_mode="resident",
            seed=17,
            out_dir=str(out / "designs" / method),
            targets=[dict(name=f"hl60_{method}", target="HL-60", method=method)],
        )
        if method == "ism":
            settings.update(rounds=3, top_k=1, ism_stride=1)
        elif method == "adalead":
            settings.update(rounds=6, population_size=32, model_queries_per_batch=256)
        elif method == "gradient":
            settings.update(rounds=50, gradient_learning_rate=0.1)
        else:
            settings.update(
                diffusion_checkpoint=str(out / "mdlm/best.pt"),
                diffusion_samples=32,
                diffusion_steps=64,
                diffusion_guidance=1.5,
                diffusion_activities=profiles["q75"],
            )
        design = DesignConfig.model_validate(settings)
        (out / f"{method}.yaml").write_text(yaml.safe_dump(design.model_dump(exclude_unset=True)))
    return training


def design(args, out):
    import regulonado.design.run as runner
    import torch
    from regulonado.config.models import DesignConfig

    original_load = runner._load_folds
    reference = DesignConfig.model_validate(yaml.safe_load((out / "ism.yaml").read_text()))
    ensemble, holdout = original_load(reference)
    original_predict = ensemble.predict
    counts = dict(calls=0, sequences=0, differentiable_sequences=0)

    def counted(batch):
        counts["calls"] += 1
        counts["sequences"] += len(batch)
        if isinstance(batch, torch.Tensor) and batch.requires_grad:
            counts["differentiable_sequences"] += len(batch)
        return original_predict(batch)

    ensemble.predict = counted
    runner._load_folds = lambda config: (ensemble, holdout)
    performance = {}
    try:
        for method in METHODS:
            marker = out / "designs" / method / "complete.json"
            if marker.exists():
                performance[method] = json.loads(marker.read_text())
                continue
            logging.info("DESIGN %s: starting all 20 test cores", method)
            counts.update(calls=0, sequences=0, differentiable_sequences=0)
            started = time.perf_counter()
            runner.run_design(
                DesignConfig.model_validate(yaml.safe_load((out / f"{method}.yaml").read_text()))
            )
            performance[method] = dict(seconds=time.perf_counter() - started, **counts)
            save_json(marker, performance[method])
            logging.info("DESIGN %s complete: %s", method, performance[method])
    finally:
        runner._load_folds = original_load
        ensemble.predict = original_predict
    save_json(out / "performance.json", performance)
    return ensemble


def evaluate(args, out, oracle=None):
    import pyfaidx
    import torch
    from regulonado.design.diffusion import (
        condition_vector,
        encode,
        load_checkpoint,
        mdlm_loss,
        sample,
    )
    from regulonado.design.region_predictor import RegionSequencePredictor
    from regulonado.genomics import one_hot, one_hot_context, window_for_interval

    if oracle is None:
        oracle = RegionSequencePredictor(args.checkpoint, device=args.device, batch_size=8)
    else:
        oracle = oracle._predictors[0]
    groups = oracle.group_names
    target = groups.index("HL-60")
    other = [i for i in range(len(groups)) if i != target]
    labels = pd.read_csv(out / "all_labels.tsv", sep="\t")
    regions = pd.read_csv(out / "selected_regions.tsv", sep="\t").set_index("name")
    rows, controls = [], []
    rng = np.random.default_rng(981)
    with pyfaidx.Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True) as fasta:
        for row in labels.loc[labels.split.eq("test")].itertuples():
            region = regions.loc[row.name]
            window = window_for_interval(
                region.chrom,
                int(region.start),
                int(region.end),
                context_length=4096,
                n_pred_bins=1,
                bin_size=1152,
                crop_bp=1408,
                snap_bp=128,
            )
            context = one_hot_context(fasta, window, 4096, len(fasta[region.chrom]))
            editable = slice(
                int(region.core_start) - window.ctx_start, int(region.core_end) - window.ctx_start
            )
            baseline = oracle(context[None])[0, :, 0].float().cpu().numpy()
            for method in METHODS:
                path = out / "designs" / method
                report = pd.read_csv(path / "designs.tsv", sep="\t").set_index("name").loc[row.name]
                # Read the actual winning FASTA, not the last generated trajectory row.
                sequences = {}
                header = None
                for line in (path / "designs.fa").read_text().splitlines():
                    if line.startswith(">"):
                        header = line[1:].split()[0].removesuffix(f"_{method}")
                    else:
                        sequences[header] = sequences.get(header, "") + line.strip()
                sequence = sequences.get(row.name)
                if sequence is None:
                    raise ValueError(
                        f"Missing final FASTA record for {row.name}: {list(sequences)[:2]}"
                    )
                candidate = context.copy()
                candidate[:, editable] = one_hot(sequence)
                scores = oracle(candidate[None])[0, :, 0].float().cpu().numpy()
                gains = scores - baseline
                rows.append(
                    dict(
                        name=row.name,
                        method=method,
                        sequence=sequence,
                        target_gain=float(gains[target]),
                        worst_offtarget_gain=float(gains[other].max()),
                        mean_offtarget_gain=float(gains[other].mean()),
                        final_target=float(scores[target]),
                        specificity=float(scores[target] - scores[other].mean()),
                        n_edits=int(report.n_edits),
                        gc=(sequence.count("G") + sequence.count("C")) / 200,
                        energy=float(report.energy),
                    )
                )
                # Random mutation controls match the winner's edit count at the same locus.
                for replicate in range(3):
                    seed_sequence = np.array(list(row.sequence))
                    positions = rng.choice(200, size=int(report.n_edits), replace=False)
                    for position in positions:
                        seed_sequence[position] = rng.choice(
                            [b for b in "ACGT" if b != seed_sequence[position]]
                        )
                    mutant = "".join(seed_sequence)
                    candidate[:, editable] = one_hot(mutant)
                    score = oracle(candidate[None])[0, :, 0].float().cpu().numpy()
                    gain = score - baseline
                    controls.append(
                        dict(
                            name=row.name,
                            method=method,
                            replicate=replicate,
                            target_gain=float(gain[target]),
                            worst_offtarget_gain=float(gain[other].max()),
                            mean_offtarget_gain=float(gain[other].mean()),
                            n_edits=int(report.n_edits),
                        )
                    )
    pd.DataFrame(rows).to_csv(out / "evaluation.tsv", sep="\t", index=False)
    pd.DataFrame(controls).to_csv(out / "random_controls.tsv", sep="\t", index=False)
    logging.info("Evaluated all final designs and edit-matched random controls")
    model, metadata = load_checkpoint(out / "mdlm/best.pt", args.device)
    profiles = json.loads((out / "conditioning_profiles.json").read_text())
    graded = []
    with pyfaidx.Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True) as fasta:
        for level, profile in profiles.items():
            values, known = condition_vector(metadata, profile)
            sequences = sample(
                model, values, known, length=200, count=32, steps=64, guidance=1.5, seed=201
            )
            for region in regions.loc[regions.split.eq("test")].itertuples():
                window = window_for_interval(
                    region.chrom,
                    region.start,
                    region.end,
                    context_length=4096,
                    n_pred_bins=1,
                    bin_size=1152,
                    crop_bp=1408,
                    snap_bp=128,
                )
                context = one_hot_context(fasta, window, 4096, len(fasta[region.chrom]))
                editable = slice(
                    region.core_start - window.ctx_start, region.core_end - window.ctx_start
                )
                for first in range(0, len(sequences), 8):
                    batch = np.repeat(context[None], len(sequences[first : first + 8]), axis=0)
                    for i, seq in enumerate(sequences[first : first + 8]):
                        batch[i, :, editable] = one_hot(seq)
                    scores = oracle(batch)[:, :, 0].float().cpu().numpy()
                    for i, score in enumerate(scores):
                        graded.append(
                            dict(
                                level=level,
                                requested_specificity=profile["HL60_specificity"],
                                name=region.Index,
                                sample=first + i,
                                target=float(score[target]),
                                contrast=float(score[target] - score[other].mean()),
                                sequence=sequences[first + i],
                            )
                        )
            logging.info("Graded specificity evaluation: %s complete", level)
    pd.DataFrame(graded).to_csv(out / "graded_generation.tsv", sep="\t", index=False)
    # Paired held-out denoising ablation: identical corruptions, true vs shuffled vs no conditions.
    val = labels.loc[labels.split.eq("validation")]
    conditions = metadata["config"]["activity_columns"]
    raw = val[conditions].to_numpy(dtype=float)
    known = np.isfinite(raw)
    values = np.where(known, (raw - np.array(metadata["mean"])) / np.array(metadata["scale"]), 0)
    ablations = []
    permutation = np.random.default_rng(82).permutation(len(val))
    for mode in ["true", "shuffled", "none"]:
        total = 0.0
        generator = torch.Generator().manual_seed(852)
        for first in range(0, len(val), 64):
            indices = np.arange(first, min(first + 64, len(val)))
            source = permutation[indices] if mode == "shuffled" else indices
            clean = torch.stack([encode(s) for s in val.sequence.iloc[indices]]).to(args.device)
            activity = torch.tensor(values[source], dtype=torch.float32, device=args.device)
            observed = torch.tensor(known[source], dtype=torch.bool, device=args.device)
            if mode == "none":
                observed[:] = False
            with torch.no_grad():
                total += float(
                    mdlm_loss(model, clean, activity, observed, generator=generator)
                ) * len(indices)
        ablations.append(dict(conditioning=mode, validation_loss=total / len(val)))
    save_json(out / "conditioning_ablation.json", ablations)
    report(out)


def report(out):
    designs = pd.read_csv(out / "evaluation.tsv", sep="\t")
    controls = pd.read_csv(out / "random_controls.tsv", sep="\t")
    controls.groupby("method").agg(
        target_gain_median=("target_gain", "median"),
        worst_offtarget_gain_median=("worst_offtarget_gain", "median"),
    ).to_csv(out / "control_summary.tsv", sep="\t")
    summary = designs.groupby("method").agg(
        target_gain_median=("target_gain", "median"),
        worst_offtarget_gain_median=("worst_offtarget_gain", "median"),
        edits_median=("n_edits", "median"),
        target_gain_positive=("target_gain", lambda x: int((x > 0).sum())),
    )
    summary.to_csv(out / "method_summary.tsv", sep="\t")
    graded = pd.read_csv(out / "graded_generation.tsv", sep="\t")
    graded.groupby("level").agg(
        target_median=("target", "median"), contrast_median=("contrast", "median")
    ).to_csv(out / "graded_summary.tsv", sep="\t")
    # Reserve slots per method and remove orientation-equivalent / near-identical candidates.
    selected = []
    for method in METHODS:
        for row in designs.loc[designs.method.eq(method)].sort_values("energy").itertuples():
            if all(
                canonical(row.sequence) != canonical(other.sequence)
                and min(
                    sum(a != b for a, b in zip(row.sequence, other.sequence)),
                    sum(
                        a != b
                        for a, b in zip(
                            row.sequence,
                            other.sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1],
                        )
                    ),
                )
                >= 10
                for other in selected
            ):
                selected.append(row)
                if sum(x.method == method for x in selected) == 4:
                    break
    pd.DataFrame([row._asdict() for row in selected]).drop(columns="Index").to_csv(
        out / "panel.tsv", sep="\t", index=False
    )
    (out / "panel.fa").write_text(
        "".join(f">{row.method}_{row.name}\n{row.sequence}\n" for row in selected)
    )
    training = json.loads((out / "mdlm/training_summary.json").read_text())
    text = [
        "# HL-60 full run",
        "",
        f"Training: {training}",
        "",
        summary.to_string(),
        "",
        (
            "Interpretation: one oracle and 20 held-out genomic loci. "
            "These scores do not establish reporter activity."
        ),
        (
            "The methods use different compute budgets; see performance.json. "
            "Numeric BED specificity and oracle contrast have different scales."
        ),
        (
            "Source-region measurements supervise 200 bp cores in broader contexts; "
            "this remains a modelling limitation."
        ),
        (
            "Graded generation and conditioning ablation test whether conditioning has learned "
            "useful control; inspect those before using diffusion designs."
        ),
    ]
    (out / "REPORT.md").write_text("\n".join(text) + "\n")
    logging.info("REPORT\n%s", summary.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "local/hl60_full_2026-11-02")
    parser.add_argument(
        "--stage", choices=["all", "prepare", "train", "design", "evaluate"], default="all"
    )
    parser.add_argument(
        "--fasta", type=Path, default=Path("reference/hg38.fa")
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=ROOT / "models_from_ccb/region_counts/dataset/regions.parquet",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "models_from_ccb/regions_more_specific_to_HL60.bed",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models_from_ccb/alphagenome-encoder"
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ["all", "prepare"]:
        prepare(args, out)
    training = configs(args, out)
    if args.stage in ["all", "train"]:
        from regulonado.design.diffusion import train_diffusion

        train_diffusion(out / "training.tsv", out / "mdlm", training)
    ensemble = None
    if args.stage in ["all", "design"]:
        ensemble = design(args, out)
    if args.stage in ["all", "evaluate"]:
        evaluate(args, out, ensemble)


if __name__ == "__main__":
    main()
