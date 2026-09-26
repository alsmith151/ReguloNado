"""Independent, staged training runs: each run follows its recipe's phases in order.

One ``train_phase`` rule serves every run. What a run's trunk and target need is
decided per run: ``trunk: live`` runs train on the profile dataset (after a CPU
schedule preflight); ``trunk: cached`` runs train a head on the region-count dataset
and their trunk's embedding cache.
"""

import re
import shlex

from regulonado.training.overrides import hydra_override_items, merge_training_settings

LIVE_RUN_NAMES = [run["name"] for run in RUNS if run.get("trunk", "live") == "live"]


def _is_cached(run_name):
    return RUN_BY_NAME[run_name].get("trunk", "live") == "cached"


def _override_flags(wildcards):
    """Render merged settings as repeatable, shell-safe CLI overrides."""
    phase = run_phase(wildcards.run, wildcards.phase)
    run = RUN_BY_NAME[wildcards.run]
    cached = _is_cached(wildcards.run)
    merged = merge_training_settings(
        [TRAIN.get("common", {}), phase.get("settings", {}), run.get("settings", {})],
        seed=run["seed"],
        # A cached trunk is fixed by its embedding cache; only a live trunk is built here.
        pretrained_model=None if cached else run["backbone"]["pretrained"],
        backbone_type=None if cached else run["backbone"]["type"],
    )
    if cached:
        if REGION_COUNTS.get("exclude_regions"):
            merged.setdefault("data.exclude_regions", str(REGION_COUNTS["exclude_regions"]))
    elif merged.get("trainer.specificity_panel_path"):
        merged.setdefault("trainer.specificity_panel_fasta", config["inputs"]["fasta"])
    merged.setdefault("trainer.wandb_project", "regulonado-training")
    merged.setdefault("trainer.wandb_group", f"{RESULTS.name}/{wildcards.run}")
    merged.setdefault("trainer.wandb_job_type", wildcards.phase)
    merged.setdefault("trainer.wandb_run_name", f"{wildcards.run}/{wildcards.phase}")

    flags = []
    for item in hydra_override_items(merged):
        flags.extend(("--set", shlex.quote(item)))
    return " ".join(flags)


def _dataset_dir(wildcards):
    return str(REGION_DATASET_DIR if _is_cached(wildcards.run) else training_dataset_dir())


def _trunk_args(wildcards):
    """The data flags that differ by trunk."""
    run = RUN_BY_NAME[wildcards.run]
    if _is_cached(wildcards.run):
        args = ["--trunk cached", f"--embeddings {shlex.quote(str(embedding_cache_dir(run['name'])))}"]
        if run.get("target_group"):
            args.append(f"--target-group {shlex.quote(str(run['target_group']))}")
        return " ".join(args)
    return f"--metadata {shlex.quote(str(training_dataset_dir() / 'tracks.parquet'))}"


def _phase_inputs(wildcards):
    if _is_cached(wildcards.run):
        return {
            "dataset": str(REGION_DATASET_DIR / "counts.parquet"),
            "embeddings": str(embedding_cache_dir(wildcards.run) / ".done"),
            "exclude_regions": (
                [REGION_COUNTS["exclude_regions"]] if REGION_COUNTS.get("exclude_regions") else []
            ),
        }
    return {
        "dataset": str(training_dataset_dir() / "README.md"),
        "metadata": str(training_dataset_dir() / "tracks.parquet"),
        "schedule": str(TRAIN_DIR / wildcards.run / wildcards.phase / "schedule.json"),
    }


if LIVE_RUN_NAMES:

    rule train_schedule_preflight:
        """Resolve a live-trunk phase's update schedule on CPU before it takes a GPU."""
        input:
            dataset=str(training_dataset_dir() / "README.md"),
            metadata=str(training_dataset_dir() / "tracks.parquet"),
        params:
            preset=lambda w: run_phase(w.run, w.phase)["preset"],
            data_dir=str(training_dataset_dir()),
            nproc=TRAIN["nproc_per_node"],
            overrides=_override_flags,
        output:
            schedule=str(TRAIN_DIR / "{run}" / "{phase}" / "schedule.json"),
        threads: 1
        resources:
            gpu=0,
            mem_mb=4000,
            runtime=10,
        wildcard_constraints:
            run="|".join(re.escape(name) for name in LIVE_RUN_NAMES),
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
        unpack(_phase_inputs),
        previous=previous_phase_state,
    params:
        preset=lambda w: run_phase(w.run, w.phase)["preset"],
        data_dir=_dataset_dir,
        trunk_args=_trunk_args,
        run_dir=lambda w: str(phase_run_dir(w.run, w.phase)),
        nproc=TRAIN["nproc_per_node"],
        overrides=_override_flags,
        prev_run_dir=lambda w: str(previous_phase_dir(w.run, w.phase) or ""),
        resolver=str(Path(workflow.basedir) / "scripts" / "resolve_checkpoint.py"),
    output:
        state=str(TRAIN_DIR / "{run}" / "{phase}" / "trainer_state.json"),
    resources:
        # One GPU per launched process, so the request always matches
        # --nproc-per-node instead of drifting from a hardcoded count.
        gpu=TRAIN["nproc_per_node"],
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
            {params.trunk_args} \
            --preset {params.preset:q} \
            --output-dir {params.run_dir:q} \
            --nproc-per-node {params.nproc} \
            {params.overrides} \
            "${{INIT_ARGS[@]}}" \
            > {log:q} 2>&1
        """
