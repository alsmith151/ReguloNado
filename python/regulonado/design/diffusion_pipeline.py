"""Pipeline entry points; W&B receives scalar metrics and safe settings only."""

import json
import os
from pathlib import Path


def fit_pipeline(params, table, manifest, out, encoder_run_dir=None):
    import yaml

    from regulonado.config.models import DiffusionStageConfig
    from regulonado.design.diffusion import train_diffusion

    stage = DiffusionStageConfig.model_validate(yaml.safe_load(Path(params).read_text()))
    metadata = json.loads(Path(manifest).read_text())
    updates = dict(
        validation_chromosomes=metadata["chromosomes"]["validation"],
        heldout_chromosomes=metadata["chromosomes"]["test"],
        sampling_weight_column="sampling_weight",
    )
    encoder_run_dir = encoder_run_dir or stage.encoder_run_dir
    if encoder_run_dir:
        from regulonado.workflow.scripts.resolve_checkpoint import resolve_checkpoint

        updates["encoder_checkpoint"] = str(resolve_checkpoint(encoder_run_dir))
    training = stage.training.model_copy(update=updates)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "training.yaml").write_text(yaml.safe_dump(training.model_dump()))
    run = None
    if stage.wandb_project:
        import wandb

        os.environ["WANDB_DISABLE_CODE"] = "true"
        os.environ["WANDB_DISABLE_GIT"] = "true"
        safe = {
            key: value
            for key, value in training.model_dump().items()
            if key
            not in {
                "encoder_checkpoint",
                "encoder_sha256",
                "activity_columns",
                "validation_chromosomes",
                "heldout_chromosomes",
                "condition_sampling_weights",
                "sampling_weight_column",
            }
        }
        safe.update(
            priority_sampling_weight=stage.priority_sampling_weight,
            priority_views=stage.priority_views,
            priority_jitter_bp=stage.priority_jitter_bp,
            priority_reverse_complement=stage.priority_reverse_complement,
            train_rows=metadata["split_counts"]["train"],
            validation_rows=metadata["split_counts"]["validation"],
        )
        identity_path = out / "wandb.json"
        identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
        run = wandb.init(
            id=identity.get("id"),
            resume="allow" if identity.get("id") else None,
            project=stage.wandb_project,
            entity=stage.wandb_entity,
            name=stage.wandb_run_name,
            config=safe,
            notes=(
                "Train/validation loss: timestep-weighted masked nucleotide cross-entropy "
                "(nats per valid base). Validation uses held-out chromosomes and fixed "
                "corruption seeds. Fixed-mask NLL is nats per masked base; accuracy is a fraction."
            ),
            settings=wandb.Settings(
                disable_code=True, disable_git=True, x_disable_meta=True, x_disable_stats=True
            ),
        )
        (out / "wandb.json").write_text(json.dumps({"id": run.id, "url": run.url}))
    try:
        result = train_diffusion(
            table,
            out,
            training,
            resume=out / "last.pt" if (out / "last.pt").exists() else None,
            on_epoch=(
                lambda entry: run.log(
                    {k: v for k, v in entry.items() if isinstance(v, (int, float))}
                )
            )
            if run
            else None,
        )
        if run:
            run.summary.update(json.loads((out / "training_summary.json").read_text()))
        return result
    finally:
        if run:
            run.finish()
