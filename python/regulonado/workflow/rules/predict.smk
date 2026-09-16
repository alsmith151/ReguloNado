"""Optional BigWig prediction from a completed final training phase."""

import shlex


PREDICTION = config.get("prediction")


if PREDICTION:
    PREDICTION_RUN = PREDICTION["run"]
    PREDICTION_DIR = RESULTS / "predictions" / PREDICTION_RUN

    def _prediction_extent_args():
        if PREDICTION.get("bed"):
            return f"--bed {shlex.quote(str(PREDICTION['bed']))}"
        return "--whole-genome"

    def _prediction_optional_args():
        args = []
        if PREDICTION.get("chromsizes"):
            args.extend(("--chromsizes", shlex.quote(str(PREDICTION["chromsizes"]))))
        if PREDICTION.get("tracks") is not None:
            args.extend(("--tracks", shlex.quote(",".join(map(str, PREDICTION["tracks"])))))
        if PREDICTION.get("device"):
            args.extend(("--device", shlex.quote(str(PREDICTION["device"]))))
        if PREDICTION.get("inverse_squash", False):
            args.append("--inverse-squash")
        return " ".join(args)

    rule predict_bigwigs:
        input:
            checkpoint_state=str(
                phase_run_dir(PREDICTION_RUN, PHASE_NAMES[-1]) / "trainer_state.json"
            ),
            dataset=str(training_dataset_dir() / "tracks.parquet"),
            fasta=config["inputs"]["fasta"],
        output:
            complete=str(PREDICTION_DIR / ".complete"),
        params:
            checkpoint_dir=str(phase_run_dir(PREDICTION_RUN, PHASE_NAMES[-1])),
            output_dir=str(PREDICTION_DIR),
            resolver=str(Path(workflow.basedir) / "scripts" / "resolve_checkpoint.py"),
            extent_args=_prediction_extent_args(),
            optional_args=_prediction_optional_args(),
            rtol=PREDICTION["rtol"],
            batch_size=PREDICTION["batch_size"],
        resources:
            gpu=1,
        log:
            str(RESULTS / "logs" / f"predict_{PREDICTION_RUN}.log"),
        shell:
            r"""
            set -euo pipefail
            mkdir -p {params.output_dir:q} "$(dirname {log:q})"
            CKPT=$(python {params.resolver:q} {params.checkpoint_dir:q})
            regulonado predict "$CKPT" {input.fasta:q} {params.output_dir:q} \
                --dataset {input.dataset:q} \
                {params.extent_args} \
                --rtol {params.rtol} \
                --batch-size {params.batch_size} \
                {params.optional_args} \
                > {log:q} 2>&1
            touch {output.complete:q}
            """
