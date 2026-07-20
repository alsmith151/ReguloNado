"""Snakemake workflow integration for the Regulonado CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer


def _profile_values(profile: Path | None) -> dict:
    if profile is None:
        return {}
    config_path = profile / "config.yaml" if profile.is_dir() else profile
    if not config_path.exists():
        raise typer.BadParameter(f"Snakemake profile/config not found: {config_path}")
    try:
        import yaml

        return yaml.safe_load(config_path.read_text()) or {}
    except ImportError as exc:
        raise typer.BadParameter("PyYAML is required to read a Snakemake profile") from exc


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


def pipeline(
    configfile: Annotated[
        Path, typer.Option("--configfile", "-f", help="Workflow YAML config.")
    ] = Path("config/config.yaml"),
    cores: Annotated[int, typer.Option("--cores", "-c", min=1, help="Local execution cores.")] = 1,
    jobs: Annotated[
        Optional[int], typer.Option("--jobs", "-j", min=1, help="Maximum remote jobs.")
    ] = None,
    config: Annotated[
        Optional[list[str]], typer.Option("--config", help="Override config with KEY=VALUE.")
    ] = None,
    profile: Annotated[
        Optional[Path],
        typer.Option("--profile", help="Snakemake profile directory or config.yaml."),
    ] = None,
    cluster_config: Annotated[
        Optional[Path], typer.Option("--cluster-config", help="Additional YAML resource config.")
    ] = None,
    executor: Annotated[
        str, typer.Option("--executor", help="Snakemake executor (local, slurm, ...).")
    ] = "local",
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

    profile_values = _profile_values(profile)
    cluster_values = _profile_values(cluster_config)
    profile_jobs = profile_values.get("jobs") or profile_values.get("max-jobs")
    max_jobs = jobs or profile_jobs
    overrides = _key_value_config(config or [])
    default_resources = {}
    for source in (
        profile_values.get("default-resources"),
        cluster_values.get("default-resources"),
    ):
        if isinstance(source, dict):
            default_resources.update(source)

    snakefile = Path(__file__).resolve().parents[1] / "workflow" / "Snakefile"
    resource_settings = ResourceSettings(
        cores=cores if executor == "local" else None,
        local_cores=cores,
        nodes=max_jobs if executor != "local" else None,
        default_resources=default_resources or None,
    )
    output_settings = OutputSettings(
        dryrun=dry_run,
        printshellcmds=True,
        show_failed_logs=True,
    )
    dag_settings = DAGSettings(targets=frozenset({target}) if target else frozenset())

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
            )
            dag_api = workflow.dag(dag_settings)
            if unlock:
                dag_api.unlock()
            if dag:
                dag_api.printdag()
                return
            if rulegraph:
                dag_api.printrulegraph()
                return
            dag_api.execute_workflow(
                executor=executor,
                execution_settings=ExecutionSettings(
                    lock=not unlock,
                    keep_going=keep_going,
                ),
            )
            if report is not None:
                from snakemake.report.html_reporter import ReportSettings

                dag_api.create_report(report_settings=ReportSettings(path=report))
    except Exception as exc:
        typer.echo(f"Snakemake workflow failed: {exc}", err=True)
        raise typer.Exit(1) from exc
