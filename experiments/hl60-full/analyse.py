"""Paired specificity diagnostics and plots for the completed full run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_mean(values, seed=17):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return dict(
        mean=float(values.mean()),
        lower=float(np.quantile(draws, 0.025)),
        upper=float(np.quantile(draws, 0.975)),
        n=len(values),
    )


def specificity_ablation(out, device):
    import torch
    from regulonado.design.diffusion import encode, load_checkpoint, mdlm_loss

    model, metadata = load_checkpoint(out / "mdlm/best.pt", device)
    val = pd.read_csv(out / "all_labels.tsv", sep="\t")
    val = val.loc[val.split.eq("validation") & val.HL60_specificity.notna()].copy()
    names = metadata["config"]["activity_columns"]
    raw = val[names].to_numpy(dtype=float)
    known = np.isfinite(raw)
    values = np.where(known, (raw - np.array(metadata["mean"])) / np.array(metadata["scale"]), 0.0)
    column = names.index("HL60_specificity")
    clean = torch.stack([encode(sequence) for sequence in val.sequence]).to(device)
    permutation = np.random.default_rng(41).permutation(len(val))
    results = []
    for replicate in range(8):
        for mode in ["true", "specificity_shuffled", "specificity_missing"]:
            activity, observed = values.copy(), known.copy()
            if mode == "specificity_shuffled":
                activity[:, column] = activity[permutation, column]
            if mode == "specificity_missing":
                observed[:, column] = False
            with torch.no_grad():
                loss = float(
                    mdlm_loss(
                        model,
                        clean,
                        torch.tensor(activity, dtype=torch.float32, device=device),
                        torch.tensor(observed, dtype=torch.bool, device=device),
                        generator=torch.Generator().manual_seed(121 + replicate),
                    )
                )
            results.append(dict(replicate=replicate, mode=mode, loss=loss))
    pd.DataFrame(results).to_csv(out / "specificity_ablation.tsv", sep="\t", index=False)
    return results


def analyse(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    designs = pd.read_csv(out / "evaluation.tsv", sep="\t")
    controls = pd.read_csv(out / "random_controls.tsv", sep="\t")
    graded = pd.read_csv(out / "graded_generation.tsv", sep="\t")
    history = pd.read_json(out / "mdlm/history.json")
    paired = graded.groupby(["sample", "level"])[["target", "contrast"]].mean().unstack("level")
    diagnostics = {
        metric: bootstrap_mean(paired[metric]["q75"] - paired[metric]["q25"])
        for metric in ["target", "contrast"]
    }
    sequences = graded.drop_duplicates(["level", "sample"]).pivot(
        index="sample", columns="level", values="sequence"
    )
    distances = [sum(a != b for a, b in zip(row.q25, row.q75)) for row in sequences.itertuples()]
    diagnostics["mean_q25_q75_hamming"] = float(np.mean(distances))
    diagnostics["unique_generated_sequences"] = graded.groupby("level").sequence.nunique().to_dict()
    random = controls.groupby(["name", "method"]).target_gain.mean().rename("random_target_gain")
    comparison = designs.join(random, on=["name", "method"])
    diagnostics["target_gain_vs_random"] = {
        method: bootstrap_mean(group.target_gain - group.random_target_gain)
        for method, group in comparison.groupby("method")
    }
    (out / "paired_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    ablation = pd.read_csv(out / "specificity_ablation.tsv", sep="\t").pivot(
        index="replicate", columns="mode", values="loss"
    )
    diagnostics["true_condition_loss_advantage_vs_shuffled"] = bootstrap_mean(
        ablation.specificity_shuffled - ablation.true
    )
    (out / "paired_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(history.epoch, history.train_loss, label="Training (weighted)")
    axes[0, 0].plot(history.epoch, history.validation_loss, label="Validation")
    axes[0, 0].set(
        xlabel="Epoch",
        ylabel="MDLM objective (nats/valid base)",
        title="Weighted training / uniform validation",
    )
    axes[0, 0].legend()
    for method, group in designs.groupby("method"):
        axes[0, 1].scatter(group.worst_offtarget_gain, group.target_gain, label=method, alpha=0.8)
    axes[0, 1].axhline(0, color="grey", lw=0.6)
    axes[0, 1].axvline(0, color="grey", lw=0.6)
    axes[0, 1].set(
        xlabel="Largest off-target gain",
        ylabel="HL-60 gain",
        title="Final designs: oracle predictions",
    )
    axes[0, 1].legend(fontsize=8)
    methods = list(diagnostics["target_gain_vs_random"])
    means = [diagnostics["target_gain_vs_random"][m]["mean"] for m in methods]
    lowers = [
        means[i] - diagnostics["target_gain_vs_random"][m]["lower"] for i, m in enumerate(methods)
    ]
    uppers = [
        diagnostics["target_gain_vs_random"][m]["upper"] - means[i] for i, m in enumerate(methods)
    ]
    axes[1, 0].errorbar(methods, means, yerr=[lowers, uppers], fmt="o", capsize=4)
    axes[1, 0].axhline(0, color="grey", lw=0.6)
    axes[1, 0].set(
        ylabel="HL-60 gain minus random control", title="Paired by locus; edit counts matched"
    )
    for _, group in paired["contrast"].iterrows():
        axes[1, 1].plot([0, 1, 2], group[["q25", "q50", "q75"]], alpha=0.2, color="steelblue")
    axes[1, 1].plot(
        [0, 1, 2], paired["contrast"][["q25", "q50", "q75"]].mean(), color="black", marker="o"
    )
    axes[1, 1].set(
        xticks=[0, 1, 2],
        xticklabels=["q25", "q50", "q75"],
        xlabel="Requested specificity",
        ylabel="Mean oracle contrast across test loci",
        title="Paired generation; off-target conditions fixed",
    )
    fig.savefig(out / "results.png", dpi=160)
    plt.close(fig)
    report = (out / "REPORT.md").read_text()
    report += "\n## Paired diagnostics\n\n" + json.dumps(diagnostics, indent=2) + "\n"
    effect = diagnostics["contrast"]
    if effect["lower"] <= 0 <= effect["upper"]:
        report += (
            "\nThe specificity sweep does not establish increased oracle contrast "
            "at higher conditioning.\n"
        )
    elif effect["mean"] < 0:
        report += "\nHigher requested specificity decreased oracle contrast in this paired sweep.\n"
    else:
        report += (
            "\nHigher requested specificity increased oracle contrast in this paired sweep; "
            "this is not experimental validation.\n"
        )
    (out / "REPORT.md").write_text(report)
    print(json.dumps(diagnostics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("local/hl60_full_2026-11-02"))
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    specificity_ablation(args.out, args.device)
    analyse(args.out)


if __name__ == "__main__":
    main()
