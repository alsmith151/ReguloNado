"""Staged fine-tuning rules.

Training runs as a sequence of phases. Each progressively unfreezes more of the
backbone and warm-starts from the previous phase's best checkpoint, so the phases
form a strict chain.

The chain is expressed through file dependencies: phase N declares phase N-1's
``trainer_state.json`` as an input. Snakemake derives the ordering from that, so
there is no manual job-dependency bookkeeping and a partially-completed pipeline
resumes correctly just by re-running.
"""


rule train_phase:
    """Run one fine-tuning phase.

    One unit of work: train a single Hydra experiment config to completion,
    writing checkpoints and ``trainer_state.json`` into the phase's run directory.

    The first phase trains from the pretrained backbone. Every later phase
    warm-starts from the best checkpoint of its predecessor, resolved at runtime
    by workflow/scripts/resolve_checkpoint.py — the path cannot be known when the
    DAG is planned, because it depends on which checkpoint scored best.
    """
    input:
        dataset=str(training_dataset_dir() / "dataset_info.json"),
        metadata=str(ENRICHED_METADATA),
        previous=previous_phase_state,
    params:
        experiment=lambda w: PHASES[phase_index(w.phase)]["experiment"],
        data_dir=str(training_dataset_dir()),
        run_dir=lambda w: str(phase_run_dir(w.phase)),
        wandb_project=config["train"]["wandb_project"],
        nproc=config["train"]["nproc_per_node"],
        # Empty for the first phase; otherwise resolve the predecessor's best
        # checkpoint. Kept as a shell-level assignment rather than a params
        # lambda so it is evaluated when the rule runs, not when the DAG is built.
        prev_run_dir=lambda w: (
            "" if phase_index(w.phase) == 0
            else str(phase_run_dir(PHASE_NAMES[phase_index(w.phase) - 1]))
        ),
        resolver=str(Path(workflow.basedir) / "scripts" / "resolve_checkpoint.py"),
    output:
        state=str(TRAIN_DIR / "{phase}" / "trainer_state.json"),
    log:
        str(RESULTS / "logs" / "train_{phase}.log"),
    shell:
        r"""
        set -euo pipefail

        INIT_ARG=""
        if [ -n "{params.prev_run_dir}" ]; then
            CKPT=$(python {params.resolver} "{params.prev_run_dir}")
            echo "Warm-starting {wildcards.phase} from: $CKPT"
            INIT_ARG="trainer.init_weights_from_checkpoint=$CKPT"
        fi

        regulonado train \
            --experiment {params.experiment} \
            --data-dir {params.data_dir} \
            --run-dir {params.run_dir} \
            --wandb-project {params.wandb_project} \
            --nproc-per-node {params.nproc} \
            $INIT_ARG \
            > {log} 2>&1
        """
