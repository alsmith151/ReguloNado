"""Train and evaluate a conditional denoiser based on the user's trained encoder."""

from __future__ import annotations

import argparse
import json
import logging
import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def prepare(out, baseline):
    out.mkdir(parents=True, exist_ok=True)
    for name in [
        "selected_regions.tsv",
        "anchors.bed",
        "candidates.bed",
        "core_attribution.tsv",
        "priority_regions.tsv",
        "specificity_calibration.json",
        "ism.yaml",
    ]:
        shutil.copy2(baseline / name, out / name)
    source = json.loads((baseline / "manifest.json").read_text())
    manifest = dict(
        source,
        methods=["diffusion"],
        architecture="alphagenome_encoder",
        trainable_parameters=659140,
        frozen_parameters=89957632,
        comparison_note=(
            "Same sequences and split as baseline; encoder/decoder and summary conditioning differ"
        ),
        concurrent_compute_note=(
            "MPS shared with ongoing baseline comparison; wall times are not controlled benchmarks"
        ),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    labels = pd.read_csv(baseline / "all_labels.tsv", sep="\t")
    groups = json.loads((Path(source["checkpoint"]) / "config.json").read_text())["group_names"]
    others = [group for group in groups if group != "HL-60"]
    labels["offtarget_mean"] = labels[others].mean(axis=1)
    labels["offtarget_max"] = labels[others].max(axis=1)
    labels["HL60_contrast"] = labels["HL-60"] - labels.offtarget_mean
    labels.to_csv(out / "all_labels.tsv", sep="\t", index=False)
    labels.loc[labels.split.ne("test")].to_csv(out / "training.tsv", sep="\t", index=False)
    original = json.loads((baseline / "conditioning_profiles.json").read_text())
    profiles = {}
    for level, values in original.items():
        mean = sum(values[group] for group in others) / len(others)
        profiles[level] = {
            "HL-60": values["HL-60"],
            "offtarget_mean": mean,
            "offtarget_max": max(values[group] for group in others),
            "HL60_contrast": values["HL-60"] - mean,
            "HL60_specificity": values["HL60_specificity"],
        }
    (out / "conditioning_profiles.json").write_text(json.dumps(profiles, indent=2))
    from regulonado.design.diffusion import DiffusionConfig

    config = DiffusionConfig(
        architecture="alphagenome_encoder",
        encoder_checkpoint=source["checkpoint"],
        encoder_compute_dtype="float16",
        activity_columns=list(profiles["q75"]),
        max_length=200,
        width=64,
        layers=3,
        heads=4,
        batch_size=64,
        epochs=120,
        learning_rate=0.0005,
        condition_dropout=0.15,
        condition_sampling_weights={"HL60_specificity": 4.0},
        stratified_timesteps=True,
        early_stopping_patience=40,
        early_stopping_min_epochs=40,
        early_stopping_min_delta=0.0001,
        diagnostic_examples=128,
        validation_chromosomes=["chr8", "chr10"],
        device=source["device"],
        seed=17,
    )
    (out / "training.yaml").write_text(yaml.safe_dump(config.model_dump()))
    design = yaml.safe_load((baseline / "diffusion.yaml").read_text())
    design.update(
        candidates=str(out / "candidates.bed"),
        intervals=str(out / "anchors.bed"),
        out_dir=str(out / "designs/diffusion"),
        diffusion_checkpoint=str(out / "mdlm/best.pt"),
        diffusion_activities=profiles["q75"],
    )
    (out / "diffusion.yaml").write_text(yaml.safe_dump(design))
    return config, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "local/hl60_full_2026-11-02")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "local/hl60_encoder_diffusion_2026-11-02"
    )
    parser.add_argument("--stage", choices=["prepare", "train", "evaluate", "all"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out.resolve()
    if args.stage in ["prepare", "all"]:
        config, manifest = prepare(out, args.baseline.resolve())
    else:
        from regulonado.design.diffusion import DiffusionConfig

        config = DiffusionConfig.model_validate(yaml.safe_load((out / "training.yaml").read_text()))
        manifest = json.loads((out / "manifest.json").read_text())
    if args.stage in ["train", "all"]:
        from regulonado.design.diffusion import train_diffusion

        train_diffusion(out / "training.tsv", out / "mdlm", config)
    if args.stage in ["evaluate", "all"]:
        run = runpy.run_path(str(ROOT / "experiments/hl60-full/run.py"))
        run["design"].__globals__["METHODS"] = ["diffusion"]
        settings = SimpleNamespace(
            checkpoint=Path(manifest["checkpoint"]),
            fasta=Path(manifest["reference"]),
            device=manifest["device"],
        )
        oracle = run["design"](settings, out)
        run["evaluate"](settings, out, oracle)
        analysis = runpy.run_path(str(ROOT / "experiments/hl60-full/analyse.py"))
        analysis["specificity_ablation"](out, manifest["device"])
        analysis["analyse"](out)


if __name__ == "__main__":
    main()
