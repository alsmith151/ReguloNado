"""Snakemake workflow integration for the Regulonado CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer


def _profile_values(profile: Path | None) -> dict:
    if profile is None:
        return {}
    if profile.is_dir():
        # Snakemake accepts a version-qualified name alongside the plain one;
        # SeqNado's test profile uses 'config.v8+.yaml'.
        candidates = [profile / "config.yaml", *sorted(profile.glob("config.v*.yaml"))]
        config_path = next((path for path in candidates if path.exists()), profile / "config.yaml")
    else:
        config_path = profile
    if not config_path.exists():
        raise typer.BadParameter(f"Snakemake profile/config not found: {config_path}")
    try:
        import yaml

        return yaml.safe_load(config_path.read_text()) or {}
    except ImportError as exc:
        raise typer.BadParameter("PyYAML is required to read a Snakemake profile") from exc


def _deployment_settings(values: dict):
    """Translate a profile's software-deployment keys into Snakemake settings.

    Profiles shared with SeqNado carry `use-conda` / `use-apptainer` /
    `software-deployment-method`. Because we drive Snakemake through its Python
    API rather than the command line, these have to be mapped explicitly —
    otherwise a container-based preset would run in the ambient environment
    without saying so.
    """
    from snakemake.settings.types import DeploymentSettings
    from snakemake_interface_executor_plugins.settings import DeploymentMethod

    methods = set()

    declared = values.get("software-deployment-method")
    if declared is not None:
        names = [declared] if isinstance(declared, str) else list(declared)
        for name in names:
            try:
                methods.add(DeploymentMethod[str(name).upper().replace("-", "_")])
            except KeyError:
                raise typer.BadParameter(
                    f"Unknown software-deployment-method in profile: {name!r}"
                ) from None

    if values.get("use-conda"):
        methods.add(DeploymentMethod.CONDA)
    if values.get("use-apptainer") or values.get("use-singularity"):
        methods.add(DeploymentMethod.APPTAINER)

    conda_prefix = values.get("conda-prefix")
    apptainer_prefix = values.get("apptainer-prefix") or values.get("singularity-prefix")
    apptainer_args = values.get("apptainer-args") or values.get("singularity-args") or ""

    if not methods and not conda_prefix and not apptainer_prefix and not apptainer_args:
        return None

    return DeploymentSettings(
        deployment_method=frozenset(methods),
        conda_prefix=Path(conda_prefix) if conda_prefix else None,
        apptainer_prefix=Path(apptainer_prefix) if apptainer_prefix else None,
        apptainer_args=str(apptainer_args),
    )


def _executor_settings(executor: str, values: dict):
    """Build optional executor plugin settings from profile-like config values."""
    if executor != "slurm":
        return None

    try:
        from snakemake_executor_plugin_slurm import ExecutorSettings as SlurmExecutorSettings
    except ImportError as exc:
        raise typer.BadParameter(
            "The 'slurm' executor requires snakemake-executor-plugin-slurm.",
            param_hint="--executor",
        ) from exc

    merged = values.get("executor-settings")
    settings: dict[str, object] = dict(merged) if isinstance(merged, dict) else {}

    # Accept common profile key spellings used with Snakemake CLI profiles.
    aliases = {
        "slurm-logdir": "logdir",
        "slurm-keep-successful-logs": "keep_successful_logs",
        "slurm-delete-logfiles-older-than": "delete_logfiles_older_than",
        "slurm-init-seconds-before-status-checks": "init_seconds_before_status_checks",
        "slurm-status-attempts": "status_attempts",
        "slurm-requeue": "requeue",
        "slurm-no-account": "no_account",
        "slurm-reservation": "reservation",
    }
    for key, target in aliases.items():
        value = values.get(key)
        if value is not None:
            settings[target] = value

    if "logdir" in settings and settings["logdir"] is not None:
        settings["logdir"] = Path(str(settings["logdir"]))

    return SlurmExecutorSettings(**settings)


def _key_value_config(values: list[str]) -> dict[str, str]:
    config: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise typer.BadParameter(f"Config override must be KEY=VALUE, got {value!r}")
        key, item = value.split("=", 1)
        if not key:
            raise typer.BadParameter(f"Config override has an empty key: {value!r}")
        config[key] = item
    return config


def _flatten_settings(settings: dict, prefix: str = "") -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in settings.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_settings(value, name))
        else:
            flattened[name] = value
    return flattened


def _validate_training_matrix(configfile: Path) -> None:
    """Compose every configured job so bad settings fail before scheduling."""
    import yaml

    raw = yaml.safe_load(configfile.read_text()) or {}
    train = raw.get("train")
    if not isinstance(train, dict):
        return  # The workflow's JSON Schema reports structural errors.
    phases = train.get("phases")
    runs = train.get("runs")
    if not isinstance(phases, list) or not isinstance(runs, list):
        return

    try:
        from regulonado.training.compose import resolved_training_config
    except ImportError as exc:
        raise typer.BadParameter(
            "Training configuration requires regulonado[train].", param_hint="CONFIGFILE"
        ) from exc

    for phase in phases:
        if not isinstance(phase, dict) or "preset" not in phase:
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            settings: dict[str, object] = {}
            layers = (
                train.get("common", {}),
                phase.get("settings", {}),
                run.get("settings", {}),
            )
            for layer in layers:
                if isinstance(layer, dict):
                    settings.update(_flatten_settings(layer))
            if "seed" in run:
                settings["seed"] = run["seed"]
            if "pretrained_model" in run:
                settings["backbone.pretrained_name"] = run["pretrained_model"]
            overrides = [
                "data.path=/config-validation",
                "output_dir=/config-validation",
                *(
                    f"{key}={json.dumps(value, separators=(',', ':'))}"
                    for key, value in settings.items()
                ),
            ]
            try:
                resolved_training_config(str(phase["preset"]), overrides)
            except Exception as exc:
                label = f"{run.get('name', '<unnamed>')}/{phase.get('name', '<unnamed>')}"
                raise typer.BadParameter(
                    f"Invalid training settings for {label}: {exc}", param_hint="CONFIGFILE"
                ) from exc


def pipeline(
    configfile: Annotated[
        Path, typer.Argument(help="Workflow YAML config")
    ],
    cores: Annotated[int, typer.Option("--cores", "-c", min=1, help="Local execution cores.")] = 1,
    jobs: Annotated[
        Optional[int], typer.Option("--jobs", "-j", min=1, help="Maximum remote jobs.")
    ] = None,
    config: Annotated[
        Optional[list[str]], typer.Option("--config", help="Override config with KEY=VALUE.")
    ] = None,
    preset: Annotated[
        Optional[str],
        # Deliberately no '-p' short flag: 'regulonado train -p' already means a
        # training preset (head_only, deep_finetune), which is unrelated.
        typer.Option(
            "--preset",
            help=(
                "Snakemake execution preset shortcode resolved from "
                "~/.config/snakemake/ (shared with SeqNado). Run 'regulonado init' "
                "to install them."
            ),
        ),
    ] = None,
    profile: Annotated[
        Optional[Path],
        typer.Option(
            "--profile",
            help="Snakemake profile directory or config.yaml (overrides --preset).",
        ),
    ] = None,
    cluster_config: Annotated[
        Optional[Path], typer.Option("--cluster-config", help="Additional YAML resource config.")
    ] = None,
    executor: Annotated[
        Optional[str],
        typer.Option("--executor", help="Override the executor selected by the profile"),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-n", help="Build the DAG without running jobs.")
    ] = False,
    target: Annotated[
        Optional[str], typer.Option("--target", "-t", help="Specific rule or output target.")
    ] = None,
    unlock: Annotated[bool, typer.Option("--unlock", help="Remove a stale workflow lock.")] = False,
    report: Annotated[
        Optional[Path], typer.Option("--report", help="Write a Snakemake HTML report.")
    ] = None,
    dag: Annotated[bool, typer.Option("--dag", help="Print the workflow DAG and exit.")] = False,
    rulegraph: Annotated[
        bool, typer.Option("--rulegraph", help="Print the rule graph and exit.")
    ] = False,
    keep_going: Annotated[
        bool, typer.Option("--keep-going", help="Continue independent jobs after an error.")
    ] = False,
) -> None:
    """Run the packaged Snakemake workflow through Snakemake's Python API."""
    if not configfile.exists():
        raise typer.BadParameter(
            f"Workflow config not found: {configfile}", param_hint="--configfile"
        )
    _validate_training_matrix(configfile)

    try:
        from snakemake.api import (
            SnakemakeApi,
        )
        from snakemake.settings.types import (
            ConfigSettings,
            DAGSettings,
            ExecutionSettings,
            OutputSettings,
            ResourceSettings,
            StorageSettings,
        )
    except ImportError as exc:
        typer.echo(
            "Snakemake is required for 'regulonado pipeline'. Install regulonado[workflow].",
            err=True,
        )
        raise typer.Exit(127) from exc

    # An explicit --profile path wins over a --preset shortcode.
    if profile is None and preset is not None:
        from regulonado.cli.profiles import format_available_presets, resolve_profile_path

        profile = resolve_profile_path(preset)
        if profile is None:
            raise typer.BadParameter(
                f"Unknown preset {preset!r}. Available: {format_available_presets()}",
                param_hint="--preset",
            )

    profile_values = _profile_values(profile)
    cluster_values = _profile_values(cluster_config)

    if preset is not None and profile is not None:
        from regulonado.cli.profiles import warn_if_undersized

        warn_if_undersized(preset, {**profile_values, **cluster_values})

    profile_jobs = profile_values.get("jobs") or profile_values.get("max-jobs")
    selected_executor = executor or profile_values.get("executor") or "local"
    max_jobs = jobs or profile_jobs
    overrides = _key_value_config(config or [])
    default_resources = {}
    rule_resources = {}
    for source in (
        profile_values.get("default-resources"),
        cluster_values.get("default-resources"),
    ):
        if isinstance(source, dict):
            default_resources.update(source)
    for source in (
        profile_values.get("set-resources"),
        cluster_values.get("set-resources"),
    ):
        if isinstance(source, dict):
            for rule, values in source.items():
                rule_resources.setdefault(rule, {}).update(values)

    snakefile = Path(__file__).resolve().parents[1] / "workflow" / "Snakefile"
    resource_settings = ResourceSettings(
        cores=cores if selected_executor == "local" else None,
        local_cores=cores,
        nodes=max_jobs if selected_executor != "local" else None,
        default_resources=default_resources or None,
        overwrite_resources=rule_resources,
    )
    output_settings = OutputSettings(
        dryrun=dry_run,
        printshellcmds=True,
        show_failed_logs=True,
    )
    dag_settings = DAGSettings(
        targets=frozenset({target}) if target else frozenset(),
        force_incomplete=bool(profile_values.get("rerun-incomplete", False)),
    )
    # Merged so a --cluster-config can add deployment keys on top of the profile.
    deployment_settings = _deployment_settings({**profile_values, **cluster_values})
    executor_settings = _executor_settings(
        selected_executor,
        {**profile_values, **cluster_values},
    )
    retries = int(cluster_values.get("retries", profile_values.get("retries", 0)))

    try:
        with SnakemakeApi(output_settings) as api:
            workflow = api.workflow(
                snakefile=snakefile,
                workdir=Path.cwd(),
                config_settings=ConfigSettings(
                    configfiles=(configfile.resolve(),),
                    config=overrides,
                ),
                resource_settings=resource_settings,
                storage_settings=StorageSettings(),
                deployment_settings=deployment_settings,
            )
            dag_api = workflow.dag(dag_settings)
            if unlock:
                dag_api.unlock()
                return
            if dag:
                dag_api.printdag()
                return
            if rulegraph:
                dag_api.printrulegraph()
                return
            execution_kwargs = {
                "executor": "dryrun" if dry_run else selected_executor,
                "execution_settings": ExecutionSettings(
                    lock=not unlock,
                    keep_going=keep_going,
                    latency_wait=int(profile_values.get("latency-wait", 3)),
                    retries=retries,
                ),
            }
            if executor_settings is not None:
                execution_kwargs["executor_settings"] = executor_settings
            dag_api.execute_workflow(**execution_kwargs)
            if report is not None:
                from snakemake.report.html_reporter import ReportSettings

                dag_api.create_report(report_settings=ReportSettings(path=report))
    except Exception as exc:
        typer.echo(f"Snakemake workflow failed: {exc}", err=True)
        raise typer.Exit(1) from exc
