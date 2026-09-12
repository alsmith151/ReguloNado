"""Build a workflow config, interactively or from flags.

Modelled on `seqnado config`: sequential prompts, defaults shown in brackets,
each answer validated as it is given. The same code path runs non-interactively
(every prompt resolves to its default), which is what CI and scripts use.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

from regulonado.config.genomes import GenomeEntry, load_genome_registry
from regulonado.config.models import (
    BAMNADO_METHODS,
    PHASE_PRESETS,
    DatasetConfig,
    InputsConfig,
    RecompressConfig,
    RegulonadoConfig,
    ScalingConfig,
    SeqNadoProjectRef,
    TrainConfig,
    TrainPhase,
    TrainRun,
)
from regulonado.config.prompts import ask, ask_int

logger = logging.getLogger(__name__)

DEFAULT_PRETRAINED = "johahi/flashzoi-replicate-0"


def _project_defaults(paths: list[str], interactive: bool) -> list[SeqNadoProjectRef]:
    """Turn `--from-seqnado` values into project references."""
    projects: list[SeqNadoProjectRef] = []
    for value in paths:
        name, sep, path = value.partition("=")
        if not sep:
            path = name
            name = Path(path).expanduser().resolve().parent.name
        projects.append(SeqNadoProjectRef(name=name, path=path))

    if interactive and projects:
        for project in projects:
            logger.info(f"SeqNado project '{project.name}': {project.path}")
    return projects


def _prompt_projects(interactive: bool) -> list[SeqNadoProjectRef]:
    """Collect SeqNado projects one at a time."""
    projects: list[SeqNadoProjectRef] = []
    if not interactive:
        return projects

    while ask(
        "Add a SeqNado project?" if not projects else "Add another SeqNado project?",
        "no" if projects else "yes",
        is_boolean=True,
    ):
        path = ask("Path to the SeqNado output directory", is_path=True)
        default_name = Path(str(path)).expanduser().resolve().parent.name
        name = ask("Name for this project (prefixes its track names)", default_name)
        projects.append(SeqNadoProjectRef(name=str(name), path=str(path)))
    return projects


def _describe_projects(projects: list[SeqNadoProjectRef]) -> tuple[list[str], list[str]]:
    """Return the pileup methods and scales common to every project."""
    from regulonado.tracks import summarise_projects

    entries = [{"name": p.name, "path": p.path} for p in projects]
    for row in summarise_projects(entries):
        if row.get("error"):
            logger.warning(f"{row['project']}: {row['error']}")
        else:
            logger.info(
                f"{row['project']}: {row['samples']} sample(s), "
                f"{row['bigwigs']} bigwig(s), genome={row.get('genome') or 'unknown'}"
            )

    from regulonado._seqnado import is_multi_project, open_project

    methods: set[str] | None = None
    scales: set[str] | None = None
    for project in projects:
        opened = open_project(project.path)
        subs = [sub for _, sub in opened.items()] if is_multi_project(opened) else [opened]
        project_methods: set[str] = set()
        project_scales: set[str] = set()
        for sub in subs:
            project_methods.update(sub.pileup_methods)
            project_scales.update(sub.scales)
        methods = project_methods if methods is None else methods & project_methods
        scales = project_scales if scales is None else scales & project_scales

    return sorted(methods or []), sorted(scales or [])


def _select_genome(
    genome: str | None, interactive: bool
) -> GenomeEntry | None:
    registry = load_genome_registry()
    if not registry:
        if genome:
            logger.warning(
                f"No genome registry found, so '{genome}' cannot be resolved; "
                f"you will be asked for a FASTA path directly."
            )
        return None

    names = sorted(registry)
    if genome:
        if genome not in registry:
            raise ValueError(f"Genome {genome!r} not configured. Available: {', '.join(names)}")
        return registry[genome]

    if not interactive:
        return None

    chosen = ask(f"Genome? (Available: {', '.join(names)})", names[0], choices=names)
    return registry[str(chosen)]


def build_config(
    *,
    interactive: bool = True,
    project_name: str | None = None,
    genome: str | None = None,
    from_seqnado: list[str] | None = None,
    base: RegulonadoConfig | dict[str, Any] | None = None,
) -> RegulonadoConfig:
    """Assemble a config, prompting for anything not supplied.

    ``base`` seeds the defaults. It may be a validated config or a raw mapping —
    ``--fill-missing`` passes a partial file that is incomplete by definition, so
    it must not be validated before the gaps are filled.
    """
    if base is None:
        defaults: dict[str, Any] = {}
    elif isinstance(base, RegulonadoConfig):
        defaults = base.to_dict()
    else:
        defaults = dict(base)
    inputs_defaults: dict[str, Any] = defaults.get("inputs", {})
    dataset_defaults: dict[str, Any] = defaults.get("dataset", {})
    recompress_defaults: dict[str, Any] = defaults.get("recompress", {})
    scaling_defaults: dict[str, Any] = defaults.get("scaling", {})
    train_defaults: dict[str, Any] = defaults.get("train", {})

    # --- project -----------------------------------------------------------
    name = project_name or ask("Project name?", "regulonado_project", interactive=interactive)
    name = str(name).replace(" ", "_")
    results_dir = ask(
        "Results directory?",
        defaults.get("results_dir") or f"{date.today().isoformat()}_{name}",
        interactive=interactive,
    )

    # --- track sources -----------------------------------------------------
    projects = _project_defaults(from_seqnado or [], interactive)
    if not projects and not inputs_defaults.get("bigwig_dir") and interactive:
        projects = _prompt_projects(interactive)

    method = scale = None
    if projects:
        methods, scales = _describe_projects(projects)
        if methods:
            method = ask(
                "Pileup method to use?",
                "deeptools" if "deeptools" in methods else methods[0],
                choices=methods,
                interactive=interactive,
            )
        if scales:
            scale = ask(
                "Scaling variant to use?",
                "unscaled" if "unscaled" in scales else scales[0],
                choices=scales,
                interactive=interactive,
            )
        for project in projects:
            project.method = str(method) if method else project.method
            project.scale = str(scale) if scale else project.scale

    # --- genome and intervals ----------------------------------------------
    entry = _select_genome(genome, interactive)
    fasta_default = inputs_defaults.get("fasta") or (entry.fasta if entry else None)
    fasta = ask(
        "Reference FASTA?", fasta_default, is_path=bool(interactive), interactive=interactive
    )
    intervals = ask(
        "Intervals BED (column 4 is the fold label)?",
        inputs_defaults.get("intervals"),
        is_path=bool(interactive),
        interactive=interactive,
    )

    bigwig_dir = inputs_defaults.get("bigwig_dir")
    track_sheet = inputs_defaults.get("track_sheet")
    if not projects and not bigwig_dir:
        bigwig_dir = ask(
            "Directory of BigWig tracks?",
            bigwig_dir,
            is_path=bool(interactive),
            interactive=interactive,
        )

    # Non-interactive runs have no chance to supply these later, and writing the
    # string "None" into the config would only fail much further downstream.
    unanswered = [
        label
        for label, value in (("inputs.fasta", fasta), ("inputs.intervals", intervals))
        if not value
    ]
    if unanswered:
        raise ValueError(
            "No value for "
            + ", ".join(unanswered)
            + ". Supply them with --genome (for the FASTA) or by passing an existing "
            "config to --fill-missing, or run without --no-interactive."
        )

    # --- scaling -----------------------------------------------------------
    scaling_choices = ["tmm", "original", "bamnado", "anchor"]
    if len(projects) == 1:
        # Reusing SeqNado's factors only makes sense for a single project;
        # across projects they are not on a comparable scale.
        scaling_choices.append("seqnado")
    scaling_method = ask(
        "Scale-factor method?",
        scaling_defaults.get("method", "tmm"),
        choices=scaling_choices,
        interactive=interactive,
    )
    scaling = ScalingConfig(method=str(scaling_method))

    if scaling.method == "anchor":
        for field, prompt in (
            ("anchor_regions", "Anchor regions BED/parquet file?"),
            ("background_regions", "Background regions BED/parquet file?"),
            ("heldout_regions", "Held-out regions BED/parquet file (optional)?"),
        ):
            value = ask(
                prompt,
                scaling_defaults.get(field),
                is_path=bool(interactive),
                interactive=interactive,
            )
            setattr(scaling, field, str(value) if value else None)

    bam_dir = inputs_defaults.get("bam_dir")
    if scaling.method == "bamnado":
        bam_dir = ask(
            "Directory of BAM files (one per track)?",
            bam_dir,
            is_path=bool(interactive),
            interactive=interactive,
        )
        scaling.bamnado_method = str(
            ask(
                "bamnado normalisation method?",
                scaling_defaults.get("bamnado_method", "csaw-background"),
                choices=list(BAMNADO_METHODS),
                interactive=interactive,
            )
        )
        if scaling.bamnado_method == "spike-in":
            scaling.bamnado_exogenous_prefix = str(
                ask(
                    "Reference-name prefix for spike-in sequences?",
                    scaling_defaults.get("bamnado_exogenous_prefix", "spikein_"),
                    interactive=interactive,
                )
            )
    elif scaling.method == "seqnado":
        scaling.seqnado_project = projects[0].path if projects else scaling_defaults.get(
            "seqnado_project"
        )
        spikein = ask(
            "Spike-in method whose normalisation factors to reuse?",
            scaling_defaults.get("seqnado_spikein_method", "orlando"),
            interactive=interactive,
        )
        scaling.seqnado_spikein_method = str(spikein) if spikein else None

    inputs = InputsConfig(
        intervals=str(intervals),
        fasta=str(fasta),
        bigwig_dir=str(bigwig_dir) if bigwig_dir else None,
        bam_dir=str(bam_dir) if bam_dir else None,
        track_sheet=str(track_sheet) if track_sheet else None,
        seqnado_projects=projects,
    )

    # --- dataset / recompress -------------------------------------------------
    dataset = DatasetConfig(
        **{
            **dataset_defaults,
            "context_length": ask_int(
                "Input context length (bp)?",
                dataset_defaults.get("context_length", 524_288),
                interactive=interactive,
            ),
            "bin_size": ask_int(
                "Signal bin size (bp)?",
                dataset_defaults.get("bin_size", 32),
                interactive=interactive,
            ),
            "n_pred_bins": ask_int(
                "Number of prediction bins?",
                dataset_defaults.get("n_pred_bins", 6_144),
                interactive=interactive,
            ),
            "shift_max_bp": ask_int(
                "Shift augmentation buffer per side (bp, multiple of bin size)?",
                dataset_defaults.get("shift_max_bp", 64),
                interactive=interactive,
            ),
            "extract_threads": ask_int(
                "BigWig extraction threads?",
                dataset_defaults.get("extract_threads", 32),
                interactive=interactive,
            ),
            "stage_to_scratch": ask(
                "Stage inputs to node-local scratch?",
                "yes" if dataset_defaults.get("stage_to_scratch", True) else "no",
                is_boolean=True,
                interactive=interactive,
            )
            if interactive
            else dataset_defaults.get("stage_to_scratch", True),
            # Aggregating projects frequently duplicates shared inputs/controls,
            # so content dedupe is the sensible default there.
            "dedupe_tracks": dataset_defaults.get(
                "dedupe_tracks", "content" if len(projects) > 1 else "none"
            ),
        }
    )

    recompress_enabled = (
        ask(
            "Recompress the dataset for faster random reads?",
            "yes" if recompress_defaults.get("enabled", True) else "no",
            is_boolean=True,
            interactive=interactive,
        )
        if interactive
        else recompress_defaults.get("enabled", True)
    )
    recompress = RecompressConfig(**{**recompress_defaults, "enabled": bool(recompress_enabled)})

    # --- training ----------------------------------------------------------
    nproc = ask_int(
        "GPUs per node for training?",
        train_defaults.get("nproc_per_node", 1),
        interactive=interactive,
    )

    existing_phases = train_defaults.get("phases") or []
    phase_default = (
        [phase["preset"] for phase in existing_phases] if existing_phases else list(PHASE_PRESETS)
    )
    selected = (
        ask(
            "Training phases, in order?",
            ",".join(phase_default),
            choices=list(PHASE_PRESETS),
            multi_select=True,
            interactive=interactive,
        )
        if interactive
        else phase_default
    )
    if isinstance(selected, str):
        selected = [item.strip() for item in selected.split(",") if item.strip()]
    phases = [TrainPhase(name=preset, preset=preset) for preset in selected]

    runs: list[TrainRun] = []
    existing_runs = train_defaults.get("runs") or []
    if interactive:
        count = ask_int("How many independent runs?", max(1, len(existing_runs)))
        for index in range(int(count)):
            previous = existing_runs[index] if index < len(existing_runs) else {}
            run_name = ask("Run name?", previous.get("name", f"run_{index}"))
            seed = ask_int("Random seed?", previous.get("seed", index))
            pretrained = ask(
                "Pretrained model?", previous.get("pretrained_model", DEFAULT_PRETRAINED)
            )
            runs.append(
                TrainRun(name=str(run_name), seed=int(seed), pretrained_model=str(pretrained))
            )
    elif existing_runs:
        runs = [TrainRun.model_validate(run) for run in existing_runs]
    else:
        runs = [TrainRun(name="run_0", seed=0, pretrained_model=DEFAULT_PRETRAINED)]

    train = TrainConfig(
        nproc_per_node=int(nproc),
        common=train_defaults.get("common", {}),
        phases=phases,
        runs=runs,
    )

    return RegulonadoConfig(
        results_dir=str(results_dir),
        inputs=inputs,
        dataset=dataset,
        recompress=recompress,
        scaling=scaling,
        train=train,
    )
