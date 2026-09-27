from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from shlex import join as shell_join
from typing import Annotated, Optional

import typer


def sweep_train(
    config_file: Annotated[
        Path, typer.Argument(help="W&B-generated JSON file containing one sweep trial")
    ],
) -> None:
    """Bridge one W&B trial into Hydra's normal training configuration.

    ``target: region_counts`` trains a region-count head (default preset ``pretrain``);
    ``data.embeddings_dir`` then selects a cached trunk. Otherwise the trial trains the
    profile model (default preset ``head_only``).
    """
    values = json.loads(config_file.read_text())
    if not isinstance(values, dict):
        raise typer.BadParameter(
            "Sweep trial JSON must contain an object", param_hint="CONFIG_FILE"
        )
    required = {"data.path"}
    missing = sorted(required - values.keys())
    if missing:
        raise typer.BadParameter(
            f"Sweep trial is missing required parameter(s): {', '.join(missing)}",
            param_hint="CONFIG_FILE",
        )

    target = str(values.pop("target", "profile"))
    if target not in ("profile", "region_counts"):
        raise typer.BadParameter(
            f"Sweep trial target must be profile or region_counts, got {target!r}",
            param_hint="CONFIG_FILE",
        )
    preset = values.pop("preset", None)
    preset = None if preset is None else str(preset)
    embeddings = values.pop("data.embeddings_dir", None)
    if embeddings is not None and target != "region_counts":
        raise typer.BadParameter(
            "data.embeddings_dir applies to target: region_counts trials",
            param_hint="CONFIG_FILE",
        )
    dataset = Path(str(values.pop("data.path")))
    configured_output = values.pop("output_dir", None)
    # A trial always reports to its sweep, in the agent's project: the runner exports
    # trainer.wandb_project as WANDB_PROJECT, which would otherwise move the run out of it.
    values.setdefault("trainer.report_to", ["wandb"])
    if os.environ.get("WANDB_PROJECT"):
        values.setdefault("trainer.wandb_project", os.environ["WANDB_PROJECT"])
    # Sweep agents run inside a fixed CPU allocation and handle unusually large
    # sequence examples. Avoid inheriting a workstation-oriented worker count
    # or holding two prefetched batches per worker, either of which can cause
    # the scheduler/OOM killer to terminate DataLoader workers mid-run.
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    sweep_workers = max(0, min(4, allocated_cpus))
    values.setdefault("trainer.num_workers", sweep_workers)
    values.setdefault("trainer.prefetch_factor", 1 if sweep_workers > 0 else None)
    from regulonado.training.overrides import hydra_override_items

    settings = hydra_override_items(values)

    run_id = os.environ.get("WANDB_RUN_ID", "trial")
    # Beside the pipeline's <results_dir>/parameter-sweep/: the profile dataset is
    # <results_dir>/dataset, the region-count one <results_dir>/region_counts/dataset.
    results_dir = dataset.parents[1] if target == "region_counts" else dataset.parent
    output_dir = Path(configured_output or results_dir / "parameter-sweep" / "runs" / run_id)
    if target == "region_counts":
        train(
            dataset=dataset,
            trunk="cached" if embeddings is not None else "live",
            target=target,
            embeddings=None if embeddings is None else Path(str(embeddings)),
            output_dir=output_dir,
            preset=preset,
            settings=settings,
        )
        return
    preset = preset or "head_only"
    # Every sweep trial validates its real dataset-backed update budget before
    # constructing the backbone in the same allocated job.
    train(
        dataset=dataset,
        output_dir=output_dir,
        preset=preset,
        settings=settings,
        schedule_only=True,
    )
    train(
        dataset=dataset,
        output_dir=output_dir,
        preset=preset,
        settings=settings,
    )


def train(
    dataset: Annotated[
        Path,
        typer.Argument(
            help="Training dataset: the profile dataset directory (target profile), or the "
            "region-count dataset from 'counts gather' (target region_counts)"
        ),
    ],
    trunk: Annotated[
        str,
        typer.Option(
            "--trunk",
            help="live: run (and optionally fine-tune) the trunk every step; cached: train a "
            "head on embeddings cached by 'embed regions'",
        ),
    ] = "live",
    target: Annotated[
        Optional[str],
        typer.Option(
            "--target",
            help="profile or region_counts; default region_counts with --trunk cached, "
            "else profile",
        ),
    ] = None,
    embeddings: Annotated[
        Optional[Path],
        typer.Option("--embeddings", "-e", help="Embedding cache directory (trunk cached)"),
    ] = None,
    fasta: Annotated[
        Optional[Path],
        typer.Option(
            "--fasta", help="Genome FASTA for a live trunk on region counts (data.fasta)"
        ),
    ] = None,
    target_group: Annotated[
        Optional[str],
        typer.Option(
            "--target-group", help="Set data.target_group (target region_counts), e.g. HL-60"
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Run directory for checkpoints and diagnostics"),
    ] = None,
    preset: Annotated[
        Optional[str],
        typer.Option(
            "--preset",
            "-p",
            help="Named phase preset: python/configs/experiment/ (target profile, default "
            "head_only) or python/configs/region_experiment/ (target region_counts, default "
            "pretrain)",
        ),
    ] = None,
    metadata: Annotated[
        Optional[Path],
        typer.Option("--metadata", help="tracks.parquet to use instead of the dataset copy"),
    ] = None,
    nproc_per_node: Annotated[
        int,
        typer.Option(
            "--nproc-per-node",
            help="Use torchrun with this many local processes when >1",
        ),
    ] = 1,
    resume_from_checkpoint: Annotated[
        Optional[str],
        typer.Option(
            "--resume-from-checkpoint",
            help="Full Trainer resume from a checkpoint dir, or 'true' for latest in output-dir",
        ),
    ] = None,
    init_weights_from_checkpoint: Annotated[
        Optional[Path],
        typer.Option(
            "--init-weights-from-checkpoint",
            help="Warm start from model weights only with a fresh optimizer/scheduler",
        ),
    ] = None,
    max_steps: Annotated[
        Optional[int],
        typer.Option("--max-steps", help="Override trainer.max_steps"),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", help="Override per-device train batch size"),
    ] = None,
    eval_batch_size: Annotated[
        Optional[int],
        typer.Option("--eval-batch-size", help="Override per-device eval batch size"),
    ] = None,
    learning_rate: Annotated[
        Optional[float],
        typer.Option("--learning-rate", "--lr", help="Override head learning rate"),
    ] = None,
    backbone_lr: Annotated[
        Optional[float],
        typer.Option("--backbone-lr", help="Override backbone learning rate"),
    ] = None,
    num_workers: Annotated[
        Optional[int],
        typer.Option("--num-workers", help="Override DataLoader worker count"),
    ] = None,
    no_wandb: Annotated[
        bool,
        typer.Option("--no-wandb", help="Disable W&B reporting for this run"),
    ] = False,
    settings: Annotated[
        Optional[list[str]],
        typer.Option(
            "--set",
            help="Override one setting as KEY=VALUE (repeatable)",
        ),
    ] = None,
    print_config: Annotated[
        bool,
        typer.Option("--print-config", help="Print the resolved training config and exit"),
    ] = False,
    schedule_only: Annotated[
        bool,
        typer.Option(
            "--schedule-only",
            help="Read Parquet metadata, print the resolved schedule, and exit before model setup",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the resolved command without running it"),
    ] = False,
) -> None:
    """Train one model from a named preset.

    Use ``--set`` for less common settings, for example
    ``--set trainer.max_eval_samples=200``. Use ``regulonado pipeline`` when
    several independent runs or warm-start phases should be orchestrated together.
    """
    if resume_from_checkpoint and init_weights_from_checkpoint:
        typer.echo(
            "Set only one of --resume-from-checkpoint or --init-weights-from-checkpoint.",
            err=True,
        )
        raise typer.Exit(1)
    if trunk not in ("live", "cached"):
        raise typer.BadParameter("Expected 'live' or 'cached'", param_hint="--trunk")
    target = target or ("region_counts" if trunk == "cached" else "profile")
    if target not in ("profile", "region_counts"):
        raise typer.BadParameter("Expected 'profile' or 'region_counts'", param_hint="--target")
    if trunk == "cached" and target != "region_counts":
        raise typer.BadParameter(
            "--trunk cached trains region_counts heads only", param_hint="--trunk/--target"
        )
    regions = target == "region_counts"
    if (trunk == "cached") != (embeddings is not None):
        raise typer.BadParameter(
            "--embeddings is required with --trunk cached, and only valid with it",
            param_hint="--trunk/--embeddings",
        )
    if fasta is not None and not (regions and trunk == "live"):
        raise typer.BadParameter(
            "--fasta only applies to --trunk live --target region_counts", param_hint="--fasta"
        )
    profile_only = {
        "--metadata": metadata,
        "--max-steps": max_steps,
        "--eval-batch-size": eval_batch_size,
        "--schedule-only": schedule_only or None,
    }
    if trunk == "cached":
        profile_only["--backbone-lr"] = backbone_lr
    if regions:
        used = [flag for flag, value in profile_only.items() if value is not None]
        if used:
            raise typer.BadParameter(
                f"{', '.join(used)} do not apply to --target region_counts"
                + (" or --trunk cached" if trunk == "cached" else ""),
                param_hint="--target",
            )
    elif target_group is not None:
        overrides_hint = "--set data.target_group=... is not a profile-model setting"
        raise typer.BadParameter(
            f"--target-group only applies to --target region_counts ({overrides_hint})",
            param_hint="--target-group",
        )
    preset = preset or ("pretrain" if regions else "head_only")

    if regions:
        overrides = [f"+region_experiment={preset}", f"data.path={dataset}"]
        if embeddings is not None:
            overrides.append(f"data.embeddings_dir={embeddings}")
        if fasta is not None:
            overrides.append(f"data.fasta={fasta}")
        if target_group is not None:
            overrides.append(f"data.target_group={target_group}")
    else:
        overrides = [
            f"+experiment={preset}",
            f"data.path={dataset}",
        ]
    if metadata is not None:
        overrides.append(f"data.metadata_path={metadata}")
    if output_dir is not None:
        overrides.append(f"output_dir={output_dir}")
    if resume_from_checkpoint is not None:
        overrides.append(f"trainer.resume_from_checkpoint={resume_from_checkpoint}")
    if init_weights_from_checkpoint is not None:
        overrides.append(f"trainer.init_weights_from_checkpoint={init_weights_from_checkpoint}")
    if max_steps is not None:
        overrides.append(f"trainer.max_steps={max_steps}")
    if batch_size is not None:
        overrides.append(f"trainer.batch_size={batch_size}")
    if eval_batch_size is not None:
        overrides.append(f"trainer.eval_batch_size={eval_batch_size}")
    if learning_rate is not None:
        overrides.append(f"trainer.learning_rate={learning_rate}")
    if backbone_lr is not None:
        overrides.append(f"trainer.backbone_learning_rate={backbone_lr}")
    if num_workers is not None:
        overrides.append(f"trainer.num_workers={num_workers}")
    if no_wandb:
        overrides.append("trainer.report_to=[]")
    for setting in settings or []:
        if "=" not in setting or not setting.split("=", 1)[0].strip():
            raise typer.BadParameter(
                f"Setting must be KEY=VALUE, got {setting!r}", param_hint="--set"
            )
        overrides.append(setting)

    if print_config and regions:
        try:
            from regulonado.training.regions.runner import resolved_region_config
        except ImportError as exc:
            typer.echo(
                "Hydra is required to inspect training presets. Install regulonado[train].",
                err=True,
            )
            raise typer.Exit(127) from exc
        try:
            typer.echo(resolved_region_config(preset, overrides[1:]))
        except Exception as exc:
            raise typer.BadParameter(
                f"Could not compose preset {preset!r}: {exc}", param_hint="--preset/--set"
            ) from exc
        return

    if print_config or schedule_only:
        try:
            from regulonado.training.compose import resolved_training_config
        except ImportError as exc:
            typer.echo(
                "Hydra is required to inspect training presets. Install regulonado[train].",
                err=True,
            )
            raise typer.Exit(127) from exc
        try:
            rendered = resolved_training_config(preset, overrides[1:])
        except Exception as exc:
            raise typer.BadParameter(
                f"Could not compose preset {preset!r}: {exc}", param_hint="--preset/--set"
            ) from exc
        if print_config:
            typer.echo(rendered)
            return
        import yaml
        from regulonado.training.runner import preflight_training_schedule

        config = yaml.safe_load(rendered)
        schedule = preflight_training_schedule(config, world_size=nproc_per_node)
        typer.echo(json.dumps(asdict(schedule), indent=2))
        return

    module = "regulonado.training.regions.runner" if regions else "regulonado.training.runner"
    if nproc_per_node > 1:
        import random

        # Avoid port collisions when multiple jobs land on the same node.
        job_id = int(os.environ.get("SLURM_JOB_ID", 0))
        master_port = 29500 + (job_id % 1000) if job_id else random.randint(29500, 30499)
        command = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            f"--master_port={master_port}",
            "-m",
            module,
            *overrides,
        ]
    else:
        command = [sys.executable, "-m", module, *overrides]

    env = os.environ.copy()
    if nproc_per_node <= 1:
        # Pin to a single visible GPU so a single training process can never
        # see >1 device: transformers.Trainer auto-wraps nn.DataParallel
        # across every visible GPU otherwise, which crashes (StopIteration
        # on replicated submodules) and silently splits the batch wrong even
        # when it doesn't crash.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0].strip() if visible else "0"

    typer.echo(shell_join(command))
    if dry_run:
        return
    raise typer.Exit(subprocess.run(command, env=env).returncode)
