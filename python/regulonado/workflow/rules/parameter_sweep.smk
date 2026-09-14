"""W&B parameter sweep with independently scheduled one-GPU agents."""

if PARAMETER_SWEEP and PARAMETER_SWEEP.get("enabled", False):
    _SWEEP_DIR = RESULTS / "parameter-sweep"
    _SWEEP_AGENT_IDS = [str(index) for index in range(int(PARAMETER_SWEEP.get("agents", 1)))]

    rule create_parameter_sweep:
        input:
            sweep_config=lambda w: str(PARAMETER_SWEEP["sweep_config"]),
        output:
            sweep_id=str(_SWEEP_DIR / "sweep.id"),
        params:
            output_dir=str(_SWEEP_DIR),
            project=str(PARAMETER_SWEEP.get("wandb_project", "regulonado-parameter-sweep")),
        log:
            str(RESULTS / "logs" / "parameter_sweep_create.log"),
        shell:
            r"""
            set -euo pipefail
            mkdir -p {params.output_dir:q}
            wandb sweep --project {params.project:q} {input.sweep_config:q} > {log:q} 2>&1
            SWEEP_ID=$(awk '/wandb agent/ {{print $NF; exit}}' {log:q})
            test -n "$SWEEP_ID"
            printf '%s\n' "$SWEEP_ID" > {output.sweep_id:q}
            """

    rule parameter_sweep_agent:
        input:
            dataset=str(training_dataset_dir() / "dataset_dict.json"),
            metadata=str(training_dataset_dir() / "tracks.parquet"),
            sweep_id=str(_SWEEP_DIR / "sweep.id"),
        output:
            done=str(_SWEEP_DIR / "agents" / "agent_{agent}.done"),
        params:
            trials=int(PARAMETER_SWEEP.get("trials_per_agent", 1)),
            agent_dir=str(_SWEEP_DIR / "agents"),
            project=str(PARAMETER_SWEEP.get("wandb_project", "regulonado-parameter-sweep")),
        resources:
            gpu=1,
        wildcard_constraints:
            agent="|".join(_SWEEP_AGENT_IDS),
        log:
            str(RESULTS / "logs" / "parameter_sweep_agent_{agent}.log"),
        shell:
            r"""
            set -euo pipefail
            mkdir -p {params.agent_dir:q}
            SWEEP_ID=$(cat {input.sweep_id:q})
            test -n "$SWEEP_ID"
            WANDB_JOB_TYPE=parameter-sweep wandb agent \
                --project {params.project:q} \
                --forward-signals \
                --count {params.trials} \
                "$SWEEP_ID" > {log:q} 2>&1
            date -u +%FT%TZ > {output.done:q}
            """

    rule parameter_sweep:
        input:
            expand(str(_SWEEP_DIR / "agents" / "agent_{agent}.done"), agent=_SWEEP_AGENT_IDS),
        output:
            done=str(_SWEEP_DIR / "sweep.done"),
        shell:
            "date -u +%FT%TZ > {output.done:q}"

    localrules: create_parameter_sweep, parameter_sweep
