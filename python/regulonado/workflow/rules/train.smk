"""Independent, staged fine-tuning runs."""

import re
import shlex

from regulonado.training.overrides import hydra_override_items, merge_training_settings


def _override_flags(wildcards):
    """Render merged settings as repeatable, shell-safe CLI overrides."""
    phase = PHASE_BY_NAME[wildcards.phase]
    run = RUN_BY_NAME[wildcards.run]
    merged = merge_training_settings(
        [
            config["train"].get("common", {}),
            phase.get("settings", {}),
            run.get("settings", {}),
        ],
        seed=run["seed"],
        pretrained_model=run["pretrained_model"],
    )
    if merged.get("trainer.specificity_panel_path"):
        merged.setdefault("trainer.specificity_panel_fasta", config["inputs"]["fasta"])
    merged.setdefault("trainer.wandb_project", "regulonado-training")
    merged.setdefault("trainer.wandb_group", f"{RESULTS.name}/{wildcards.run}")
    merged.setdefault("trainer.wandb_job_type", wildcards.phase)
    merged.setdefault("trainer.wandb_run_name", f"{wildcards.run}/{wildcards.phase}")

    flags = []
    for item in hydra_override_items(merged):
        flags.extend(("--set", shlex.quote(item)))
    return " ".join(flags)


rule train_schedule_preflight:
    input:
        dataset=str(training_dataset_dir() / "README.md"),
        metadata=str(training_dataset_dir() / "tracks.parquet"),
    params:
        preset=lambda w: PHASE_BY_NAME[w.phase]["preset"],
        data_dir=str(training_dataset_dir()),
        nproc=config["train"]["nproc_per_node"],
        overrides=_override_flags,
    output:
        schedule=str(TRAIN_DIR / "{run}" / "{phase}" / "schedule.json"),
    threads:
        1
    resources:
        gpu=0,
        mem_mb=4000,
        runtime=10,
    wildcard_constraints:
        run="|".join(re.escape(name) for name in RUN_NAMES),
        phase="|".join(re.escape(name) for name in PHASE_NAMES),
    log:
        str(RESULTS / "logs" / "schedule_{run}_{phase}.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {output.schedule:q})" "$(dirname {log:q})"
        regulonado train \
            {params.data_dir:q} \
            --preset {params.preset:q} \
            --metadata {input.metadata:q} \
            --nproc-per-node {params.nproc} \
            {params.overrides} \
            --schedule-only \
            > {output.schedule:q} 2> {log:q}
        """


rule train_phase:
    input:
        dataset=str(training_dataset_dir() / "README.md"),
        metadata=str(training_dataset_dir() / "tracks.parquet"),
        schedule=str(TRAIN_DIR / "{run}" / "{phase}" / "schedule.json"),
        previous=previous_phase_state,
    params:
        preset=lambda w: PHASE_BY_NAME[w.phase]["preset"],
        data_dir=str(training_dataset_dir()),
        run_dir=lambda w: str(phase_run_dir(w.run, w.phase)),
        nproc=config["train"]["nproc_per_node"],
        overrides=_override_flags,
        prev_run_dir=lambda w: (
            ""
            if phase_index(w.phase) == 0
            else str(phase_run_dir(w.run, PHASE_NAMES[phase_index(w.phase) - 1]))
        ),
        resolver=str(Path(workflow.basedir) / "scripts" / "resolve_checkpoint.py"),
    output:
        state=str(TRAIN_DIR / "{run}" / "{phase}" / "trainer_state.json"),
    resources:
        # One GPU per launched process, so the request always matches
        # --nproc-per-node instead of drifting from a hardcoded count.
        gpu=config["train"]["nproc_per_node"],
    wildcard_constraints:
        run="|".join(re.escape(name) for name in RUN_NAMES),
        phase="|".join(re.escape(name) for name in PHASE_NAMES),
    log:
        str(RESULTS / "logs" / "train_{run}_{phase}.log"),
    shell:
        r"""
        set -euo pipefail

        INIT_ARGS=()
        if [ -n "{params.prev_run_dir}" ]; then
            CKPT=$(python {params.resolver:q} "{params.prev_run_dir}")
            echo "Warm-starting {wildcards.run}/{wildcards.phase} from: $CKPT"
            INIT_ARGS=(--init-weights-from-checkpoint "$CKPT")
        fi

        regulonado train \
            {params.data_dir:q} \
            --preset {params.preset:q} \
            --metadata {input.metadata:q} \
            --output-dir {params.run_dir:q} \
            --nproc-per-node {params.nproc} \
            {params.overrides} \
            "${{INIT_ARGS[@]}}" \
            > {log:q} 2>&1
        """
