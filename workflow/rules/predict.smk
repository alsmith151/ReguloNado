"""Prediction rule: trained checkpoint -> BigWig tracks.

Prediction was previously disconnected from the pipeline — there was no launcher
and no dependency edge from training, so it had to be run by hand against a
checkpoint path found manually. Here it is a real DAG node consuming the final
training phase's output.
"""


rule predict:
    """Generate BigWig predictions from the final trained model.

    One unit of work: load the best checkpoint of the last training phase, run
    the model over either a supplied BED of regions or the whole genome, and
    write one BigWig per predicted track.

    Only runs when `predict.enabled` is true, since whole-genome prediction is
    expensive.
    """
    input:
        state=str(phase_run_dir(PHASE_NAMES[-1]) / "trainer_state.json"),
        fasta=config["inputs"]["fasta"],
    params:
        run_dir=str(phase_run_dir(PHASE_NAMES[-1])),
        out_dir=str(PREDICT_DIR),
        resolver=str(Path(workflow.basedir) / "scripts" / "resolve_checkpoint.py"),
        regions=lambda w: (
            f"--regions {config['predict']['regions']}"
            if config["predict"]["regions"]
            else ""
        ),
        batch_size=config["predict"]["batch_size"],
        inverse_squash=lambda w: (
            "--inverse-squash" if config["predict"]["inverse_squash"] else "--no-inverse-squash"
        ),
    output:
        done=touch(str(PREDICT_DIR / "predictions.done")),
    log:
        str(RESULTS / "logs" / "predict.log"),
    shell:
        r"""
        set -euo pipefail
        CKPT=$(python {params.resolver} "{params.run_dir}")
        echo "Predicting from checkpoint: $CKPT"

        regulonado predict \
            "$CKPT" \
            {input.fasta} \
            {params.out_dir} \
            {params.regions} \
            --batch-size {params.batch_size} \
            {params.inverse_squash} \
            > {log} 2>&1
        """
