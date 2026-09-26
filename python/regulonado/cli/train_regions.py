from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from shlex import join as shell_join
from typing import Annotated, Optional

import typer


def train_regions(
    region_dataset: Annotated[
        Path,
        typer.Argument(help="RegionCountData run directory (regulonado counts gather)"),
    ],
    embeddings: Annotated[
        Path,
        typer.Option(
            "--embeddings", "-e", help="Embeddings cache directory (regulonado embed regions)"
        ),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Run directory for checkpoints and diagnostics"),
    ] = None,
    preset: Annotated[
        str,
        typer.Option("--preset", "-p", help="Named regions_experiment preset"),
    ] = "pretrain",
    target_group: Annotated[
        Optional[str],
        typer.Option("--target-group", help="Override data.target_group (e.g. HL-60)"),
    ] = None,
    nproc_per_node: Annotated[
        int,
        typer.Option(
            "--nproc-per-node", help="Use torchrun with this many local processes when >1"
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
            help="Warm start from model weights only (shape-mismatched tensors are skipped)",
        ),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", help="Override per-device train batch size"),
    ] = None,
    learning_rate: Annotated[
        Optional[float],
        typer.Option("--learning-rate", "--lr", help="Override head learning rate"),
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
        typer.Option("--set", help="Override one setting as KEY=VALUE (repeatable)"),
    ] = None,
    print_config: Annotated[
        bool,
        typer.Option("--print-config", help="Print the resolved region-training config and exit"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the resolved command without running it"),
    ] = False,
) -> None:
    """Train a region-count head on cached backbone embeddings from a named preset.

    Use ``--set`` for less common settings, for example ``--set data.gini_std_threshold=1.5``.
    """
    if resume_from_checkpoint and init_weights_from_checkpoint:
        typer.echo(
            "Set only one of --resume-from-checkpoint or --init-weights-from-checkpoint.",
            err=True,
        )
        raise typer.Exit(1)

    overrides = [
        f"+regions_experiment={preset}",
        f"data.path={region_dataset}",
        f"data.embeddings_dir={embeddings}",
    ]
    if output_dir is not None:
        overrides.append(f"output_dir={output_dir}")
    if target_group is not None:
        overrides.append(f"data.target_group={target_group}")
    if resume_from_checkpoint is not None:
        overrides.append(f"trainer.resume_from_checkpoint={resume_from_checkpoint}")
    if init_weights_from_checkpoint is not None:
        overrides.append(f"trainer.init_weights_from_checkpoint={init_weights_from_checkpoint}")
    if batch_size is not None:
        overrides.append(f"trainer.batch_size={batch_size}")
    if learning_rate is not None:
        overrides.append(f"trainer.learning_rate={learning_rate}")
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

    if print_config:
        try:
            from regulonado.regions.runner import resolved_regions_config
        except ImportError as exc:
            typer.echo(
                "Hydra is required to inspect region-training presets. Install regulonado[train].",
                err=True,
            )
            raise typer.Exit(127) from exc
        try:
            rendered = resolved_regions_config(preset, overrides[1:])
        except Exception as exc:
            raise typer.BadParameter(
                f"Could not compose preset {preset!r}: {exc}", param_hint="--preset/--set"
            ) from exc
        typer.echo(rendered)
        return

    if nproc_per_node > 1:
        import random

        job_id = int(os.environ.get("SLURM_JOB_ID", 0))
        master_port = 29500 + (job_id % 1000) if job_id else random.randint(29500, 30499)
        command = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            f"--master_port={master_port}",
            "-m",
            "regulonado.regions.runner",
            *overrides,
        ]
    else:
        command = [sys.executable, "-m", "regulonado.regions.runner", *overrides]

    env = os.environ.copy()
    if nproc_per_node <= 1:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0].strip() if visible else "0"

    typer.echo(shell_join(command))
    if dry_run:
        return
    raise typer.Exit(subprocess.run(command, env=env).returncode)
