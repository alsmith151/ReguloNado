"""Independent, staged fine-tuning runs."""

import json
import re
import shlex


def _flatten_settings(settings, prefix=""):
    flattened = {}
    for key, value in settings.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_settings(value, name))
        else:
            flattened[name] = value
    return flattened


def _override_flags(wildcards):
    """Render merged settings as repeatable, shell-safe CLI overrides."""
    phase = PHASE_BY_NAME[wildcards.phase]
    run = RUN_BY_NAME[wildcards.run]
    merged = {}
    for settings in (
        config["train"].get("common", {}),
        phase.get("settings", {}),
        run.get("settings", {}),
    ):
        merged.update(_flatten_settings(settings))

    # Run identity always wins over generic settings.
    merged["seed"] = run["seed"]
    merged["backbone.pretrained_name"] = run["pretrained_model"]

    flags = []
    for key, value in sorted(merged.items()):
        rendered = json.dumps(value, separators=(",", ":"))
        flags.extend(("--set", shlex.quote(f"{key}={rendered}")))
    return " ".join(flags)


rule train_phase:
    input:
        dataset=str(training_dataset_dir() / "dataset_dict.json"),
        metadata=str(ENRICHED_METADATA),
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
