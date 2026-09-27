"""W&B parameter sweeps with independently scheduled one-GPU agents.

Each enabled ``parameter_sweeps.<name>`` gets ``<results_dir>/parameter-sweeps/<name>/``:
its W&B sweep id, the agents' markers and the trials' run directories.
"""

import re

if PARAMETER_SWEEPS:
    _SWEEPS_DIR = RESULTS / "parameter-sweeps"
    # Region-count sweeps' copies of embedding caches, shared by every sweep reading the same
    # cache at the same size.
    _SWEEP_EMBEDDINGS_DIR = _SWEEPS_DIR / "embeddings"


    def _sweep(wildcards):
        return PARAMETER_SWEEPS[wildcards.sweep]


    def _sweep_agent_ids(name):
        return [str(index) for index in range(int(PARAMETER_SWEEPS[name].get("agents", 1)))]


    def _sweep_cache_dirs(name):
        """The embedding caches (or their copies) sweep *name*'s trials read."""
        sweep = PARAMETER_SWEEPS[name]
        caches = [embedding_cache_dir(run).name for run in sweep.get("embeddings_from", [])]
        subset = sweep.get("embeddings_subset")
        if subset:
            return [_SWEEP_EMBEDDINGS_DIR / f"{cache}-train{subset}" for cache in caches]
        return [EMBEDDINGS_DIR / cache for cache in caches]


    def _sweep_inputs(wildcards):
        """The dataset the trials train on, and any embedding caches they read."""
        if _sweep(wildcards).get("target", "profile") == "region_counts":
            return {
                "dataset": str(REGION_DATASET_DIR / "counts.parquet"),
                "embeddings": [
                    str(path / ".done") for path in _sweep_cache_dirs(wildcards.sweep)
                ],
            }
        return {
            "dataset": str(training_dataset_dir() / "README.md"),
            "metadata": str(training_dataset_dir() / "tracks.parquet"),
        }


    _SUBSET_CACHES = sorted(
        {
            path.name
            for name in PARAMETER_SWEEPS
            if PARAMETER_SWEEPS[name].get("embeddings_subset")
            for path in _sweep_cache_dirs(name)
        }
    )

    if _SUBSET_CACHES:

        rule sweep_embeddings_subset:
            """Copy N random train regions of one cache, read once from the full cache so
            the trials need not each read through it."""
            input:
                done=str(EMBEDDINGS_DIR / "{cache}" / ".done"),
                dataset=str(REGION_DATASET_DIR / "counts.parquet"),
            params:
                cache_dir=str(EMBEDDINGS_DIR / "{cache}"),
                dataset_dir=str(REGION_DATASET_DIR),
                out_dir=str(_SWEEP_EMBEDDINGS_DIR / "{cache}-train{regions}"),
            output:
                touch(str(_SWEEP_EMBEDDINGS_DIR / "{cache}-train{regions}" / ".done")),
            resources:
                mem_mb=scaled_mem_mb(32000),
            wildcard_constraints:
                cache="|".join(re.escape(name) for name in EMBEDDING_CACHES),
                regions=r"\d+",
            log:
                str(RESULTS / "logs" / "sweep_embeddings_subset_{cache}-train{regions}.log"),
            shell:
                r"""
                set -euo pipefail
                mkdir -p "$(dirname {log:q})"
                regulonado embed subset {params.cache_dir:q} {params.dataset_dir:q} \
                    --out {params.out_dir:q} --regions {wildcards.regions} --split train \
                    > {log:q} 2>&1
                """

    rule create_parameter_sweep:
        input:
            # Creating the W&B sweep is part of the dataset-dependent stage,
            # not merely configuration parsing.  Keep this dependency here
            # (rather than only on the agents) so the remote sweep cannot be
            # initialized while the dataset is still being built.
            unpack(_sweep_inputs),
            sweep_config=lambda w: str(_sweep(w)["sweep_config"]),
        output:
            sweep_id=str(_SWEEPS_DIR / "{sweep}" / "sweep.id"),
        params:
            output_dir=lambda w: str(_SWEEPS_DIR / w.sweep),
            project=lambda w: str(_sweep(w).get("wandb_project", "regulonado-parameter-sweep")),
        wildcard_constraints:
            sweep="|".join(re.escape(name) for name in PARAMETER_SWEEPS),
        log:
            str(RESULTS / "logs" / "parameter_sweep_{sweep}_create.log"),
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
            unpack(_sweep_inputs),
            sweep_id=str(_SWEEPS_DIR / "{sweep}" / "sweep.id"),
        output:
            done=str(_SWEEPS_DIR / "{sweep}" / "agents" / "agent_{agent}.done"),
        params:
            trials=lambda w: int(_sweep(w).get("trials_per_agent", 1)),
            agent_dir=lambda w: str(_SWEEPS_DIR / w.sweep / "agents"),
            runs_dir=lambda w: str(_SWEEPS_DIR / w.sweep / "runs"),
            fasta=config["inputs"]["fasta"],
            project=lambda w: str(_sweep(w).get("wandb_project", "regulonado-parameter-sweep")),
        threads: lambda w: int(_sweep(w).get("cpus_per_agent", 4))
        resources:
            gpu=1,
            mem_mb=lambda w: int(_sweep(w).get("mem_mb_per_agent", 64000)),
            runtime=lambda w: int(_sweep(w).get("runtime_minutes_per_agent", 240)),
        wildcard_constraints:
            sweep="|".join(re.escape(name) for name in PARAMETER_SWEEPS),
            agent=r"\d+",
        log:
            str(RESULTS / "logs" / "parameter_sweep_{sweep}_agent_{agent}.log"),
        shell:
            r"""
            set -euo pipefail
            mkdir -p {params.agent_dir:q}
            SWEEP_ID=$(cat {input.sweep_id:q})
            test -n "$SWEEP_ID"
            REGULONADO_FASTA={params.fasta:q} \
            REGULONADO_SWEEP_RUNS_DIR={params.runs_dir:q} \
            WANDB_JOB_TYPE=parameter-sweep wandb agent \
                --project {params.project:q} \
                --forward-signals \
                --count {params.trials} \
                "$SWEEP_ID" > {log:q} 2>&1
            date -u +%FT%TZ > {output.done:q}
            """

    rule parameter_sweep:
        input:
            lambda w: expand(
                str(_SWEEPS_DIR / w.sweep / "agents" / "agent_{agent}.done"),
                agent=_sweep_agent_ids(w.sweep),
            ),
        output:
            done=str(_SWEEPS_DIR / "{sweep}" / "sweep.done"),
        wildcard_constraints:
            sweep="|".join(re.escape(name) for name in PARAMETER_SWEEPS),
        shell:
            "date -u +%FT%TZ > {output.done:q}"

    localrules: create_parameter_sweep, parameter_sweep
