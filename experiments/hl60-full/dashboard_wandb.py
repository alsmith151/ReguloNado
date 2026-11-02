"""Create a W&B workspace with visible charts for the approved aggregate metrics."""

from __future__ import annotations

import json
from pathlib import Path

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "local/hl60_full_2026-11-02"
state = json.loads((out / "wandb_state.json").read_text())
groups = json.loads((ROOT / "models_from_ccb/alphagenome-encoder/config.json").read_text())[
    "group_names"
]
training_panels = [
    wr.LinePlot(
        title="Training MDLM objective (nats/valid base; specificity oversampled)",
        x="training/epoch",
        y=["training/mdlm_objective_nats_per_base"],
    ),
    wr.LinePlot(
        title="Validation MDLM objective (nats/valid base; uniform regions)",
        x="training/epoch",
        y=["validation/mdlm_objective_nats_per_base"],
    ),
]
for rate in ["15pct", "50pct", "90pct"]:
    training_panels.append(
        wr.LinePlot(
            title=f"{rate} masks: cross-entropy (nats/masked base)",
            x="training/epoch",
            y=[f"training/masked_token_nll_{rate}", f"validation/masked_token_nll_{rate}"],
        )
    )
    training_panels.append(
        wr.LinePlot(
            title=f"{rate} masks: reconstruction accuracy (fraction)",
            x="training/epoch",
            y=[
                f"training/masked_token_accuracy_{rate}",
                f"validation/masked_token_accuracy_{rate}",
            ],
        )
    )
sections = [
    ws.Section(
        name="Denoising loss and accuracy - explicit units", is_open=True, panels=training_panels
    )
]
for method in ["ism", "adalead", "gradient", "diffusion"]:
    prefix = f"design/{method}"
    axis = f"{prefix}/completed"
    panels = []
    for metric, title in [
        ("energy", "Objective energy (lower is better)"),
        ("target_score", "HL-60 activity"),
        ("target_gain", "HL-60 activity gain"),
        ("offtarget_boost", "Off-target boost penalty"),
        ("worst_offtarget_gain", "Largest off-target gain"),
        ("mean_offtarget_gain", "Mean off-target gain"),
        ("n_edits", "Number of edited bases"),
        ("target_minus_mean_offtarget", "HL-60 minus mean off-target"),
        ("target_minus_max_offtarget", "HL-60 minus strongest off-target"),
        ("flank_delta", "Genomic/neutral flank difference"),
        ("edit_penalty", "Edit penalty"),
    ]:
        panels.append(
            wr.LinePlot(
                title=title,
                x=axis,
                title_x="Completed loci",
                y=[f"{prefix}/{metric}_{stat}" for stat in ["mean", "median", "min", "max"]],
            )
        )
    panels.extend(
        [
            wr.LinePlot(
                title="Activity by cell group",
                x=axis,
                title_x="Completed loci",
                y=[f"{prefix}/group_{group}_mean" for group in groups],
            ),
            wr.LinePlot(
                title="Activity gain by cell group",
                x=axis,
                title_x="Completed loci",
                y=[f"{prefix}/group_gain_{group}_mean" for group in groups],
            ),
            wr.LinePlot(
                title="Fraction with increased HL-60 activity",
                x=axis,
                y=[f"{prefix}/positive_target_gain_fraction"],
            ),
            wr.LinePlot(
                title="Genomic and neutral scores",
                x=axis,
                y=[f"{prefix}/score_genomic_mean", f"{prefix}/score_neutral_mean"],
            ),
        ]
    )
    sections.append(
        ws.Section(name=f"{method.upper()} - full design metrics", is_open=True, panels=panels)
    )
saved = out / "wandb_workspace.json"
if saved.exists():
    workspace = ws.Workspace.from_url(json.loads(saved.read_text())["url"])
    workspace.sections = sections
else:
    workspace = ws.Workspace(
        entity=state["entity"],
        project=state["project"],
        name="HL60 full comparison metrics",
        sections=sections,
        runset_settings=ws.RunsetSettings(query="hl60-full-2026-11-02", pinned_runs=[state["id"]]),
    )
encoder_state = ROOT / "local/hl60_encoder_diffusion_2026-11-02/wandb_state.json"
if encoder_state.exists():
    encoder = json.loads(encoder_state.read_text())
    workspace.runset_settings.pinned_runs = [state["id"], encoder["id"]]
    workspace.runset_settings.query = "^(hl60-full-2026-11-02|hl60-encoder-diffusion-v2)$"
    workspace.runset_settings.regex_query = True
workspace.save()
(out / "wandb_workspace.json").write_text(
    json.dumps(
        dict(
            url=workspace.url,
            sections=len(sections),
            panels=sum(len(section.panels) for section in sections),
        ),
        indent=2,
    )
)
print(workspace.url)
