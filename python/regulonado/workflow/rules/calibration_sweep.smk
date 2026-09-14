"""Optional W&B-managed calibration sweep on a GPU Slurm job."""

if CALIBRATION_SWEEP and CALIBRATION_SWEEP.get("enabled", False):
    _SWEEP_DIR = RESULTS / "calibration-sweep"

    rule calibration_sweep:
        input:
            dataset=str(training_dataset_dir() / "dataset_dict.json"),
            metadata=str(training_dataset_dir() / "tracks.parquet"),
            sweep_config=lambda w: str(CALIBRATION_SWEEP["sweep_config"]),
        output:
            done=str(_SWEEP_DIR / "sweep.done"),
        params:
            output_dir=str(_SWEEP_DIR),
            agent_count=int(CALIBRATION_SWEEP.get("agent_count", 1)),
        resources:
            gpu=1,
        log:
            str(RESULTS / "logs" / "calibration_sweep.log"),
        shell:
            r"""
            set -euo pipefail
            mkdir -p {params.output_dir:q}
            SWEEP_ID=$(wandb sweep {input.sweep_config:q} | awk '/wandb agent/ {{print $NF; exit}}')
            test -n "$SWEEP_ID"
            wandb agent --count {params.agent_count} "$SWEEP_ID"
            date -u +%FT%TZ > {output.done:q}
            """
