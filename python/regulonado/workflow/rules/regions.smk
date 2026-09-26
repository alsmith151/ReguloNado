"""Region-count modelling on cached, frozen backbone embeddings.

Included from the Snakefile only when 'regions:' is configured, so every name
below can assume ``REGIONS`` is set. Three chained phases, each with its own
dynamic fan-out:

    region_count_track (per real track, CPU)   -> region_counts_gather -> RESULTS/regions/dataset
    region_embed_chrom (per {embedding}/{chrom}, GPU) -> region_embed_done
    region_train_phase (per {run}/{phase}, GPU), warm-started like rules/train.smk

Counting reuses the top-level ``inputs.fasta`` and the track table
``track_assemble`` produces; the real (QC-filtered, deduped) track list isn't
known at DAG-construction time, so ``region_count_track_names`` is a
``checkpoint`` -- the same pattern ``rules/design.smk``'s ``shard_candidates``
uses for a shard count that depends on its own input's real content.
"""

import re
import shlex
from pathlib import Path

from snakemake.exceptions import WorkflowError

from regulonado.normalization import read_regions
from regulonado.training.overrides import hydra_override_items, merge_training_settings

REGIONS = config["regions"]
REGIONS_DIR = RESULTS / "regions"
REGION_COUNTS_DIR = REGIONS_DIR / "counts" / "tracks"
REGION_DATASET_DIR = REGIONS_DIR / "dataset"
REGION_EMBEDDINGS_DIR = REGIONS_DIR / "embeddings"
REGION_TRAIN_DIR = REGIONS_DIR / "train"

REGION_EMBEDDINGS = REGIONS["embeddings"]
REGION_EMBEDDING_NAMES = [embedding["name"] for embedding in REGION_EMBEDDINGS]
REGION_EMBEDDING_BY_NAME = {embedding["name"]: embedding for embedding in REGION_EMBEDDINGS}

REGION_TRAIN = REGIONS.get("train")
REGION_PHASES = REGION_TRAIN["phases"] if REGION_TRAIN else []
REGION_RUNS = REGION_TRAIN["runs"] if REGION_TRAIN else []
REGION_PHASE_NAMES = [phase["name"] for phase in REGION_PHASES]
REGION_RUN_NAMES = [run["name"] for run in REGION_RUNS]
REGION_PHASE_BY_NAME = {phase["name"]: phase for phase in REGION_PHASES}
REGION_RUN_BY_NAME = {run["name"]: run for run in REGION_RUNS}


def region_phase_index(name):
    return REGION_PHASE_NAMES.index(name)


def region_phase_run_dir(run, phase):
    return REGION_TRAIN_DIR / run / phase


def region_previous_phase_state(wildcards):
    """Chain a region-training phase only to the preceding phase of the same run."""
    index = region_phase_index(wildcards.phase)
    if index == 0:
        return []
    return str(
        region_phase_run_dir(wildcards.run, REGION_PHASE_NAMES[index - 1]) / "trainer_state.json"
    )


def _region_chroms():
    """Distinct chromosomes in the configured region set, read once at parse time.

    The region set is a static input (like ``inputs.intervals``), so -- unlike the
    track list, which depends on ``track_assemble``'s QC output -- this can be read
    directly without a checkpoint.
    """
    regions = read_regions(Path(REGIONS["inputs"]["regions"]))
    seen: dict[str, None] = {}
    for chrom, _start, _end in regions:
        seen.setdefault(chrom, None)
    return sorted(seen)


REGION_CHROMS = _region_chroms()


def _region_embed_out_dir(embedding_name):
    return REGION_EMBEDDINGS_DIR / embedding_name


def _region_embed_command_flags(embedding):
    flags = []
    if embedding.get("pretrained"):
        flags += ["--pretrained", shlex.quote(str(embedding["pretrained"]))]
    if embedding.get("rc"):
        flags.append("--rc")
    if embedding.get("pool_to"):
        flags += ["--pool-to", str(embedding["pool_to"])]
    if embedding.get("context"):
        flags += ["--context", str(embedding["context"])]
    if embedding.get("stride"):
        flags += ["--stride", str(embedding["stride"])]
    if embedding.get("batch_size"):
        flags += ["--batch-size", str(embedding["batch_size"])]
    return " ".join(flags)


def _region_override_flags(wildcards):
    """Render merged region-training settings as repeatable, shell-safe CLI overrides.

    Mirrors ``rules/train.smk``'s ``_override_flags``, but for ``regulonado
    train-regions`` -- no ``backbone.pretrained_name`` root-level injection, since the
    frozen backbone is fixed by the run's named embedding, not a per-run checkpoint.
    """
    phase = REGION_PHASE_BY_NAME[wildcards.phase]
    run = REGION_RUN_BY_NAME[wildcards.run]
    merged = merge_training_settings(
        [
            REGION_TRAIN.get("common", {}),
            phase.get("settings", {}),
            run.get("settings", {}),
        ],
        seed=run["seed"],
    )
    merged.setdefault("trainer.wandb_project", "regulonado-regions-training")
    merged.setdefault("trainer.wandb_group", f"{RESULTS.name}/regions/{wildcards.run}")
    merged.setdefault("trainer.wandb_job_type", wildcards.phase)
    merged.setdefault("trainer.wandb_run_name", f"{wildcards.run}/{wildcards.phase}")

    flags = []
    for item in hydra_override_items(merged):
        flags.extend(("--set", shlex.quote(item)))
    return " ".join(flags)


def _region_counts_chrom_sizes_arg(wildcards):
    chrom_sizes = REGIONS["counts"].get("chrom_sizes")
    return f"--chrom-sizes {shlex.quote(str(chrom_sizes))}" if chrom_sizes else ""


checkpoint region_count_track_names:
    """Snapshot ``tracks.parquet``'s track names into one marker file per track.

    ``region_count_track`` fans out over these markers instead of a module-level
    track list, since the real (QC-filtered, deduped) track set isn't known until
    ``track_assemble`` has actually run.
    """
    input:
        table=str(TRACKS_DIR / "tracks.parquet"),
    output:
        directory(str(REGIONS_DIR / "counts" / "_track_names")),
    run:
        import pandas as pd

        out_dir = Path(output[0])
        out_dir.mkdir(parents=True, exist_ok=True)
        table = pd.read_parquet(input.table, columns=["track_name"])
        for track_name in table["track_name"]:
            (out_dir / f"{track_name}.track").touch()


def _region_track_names(wildcards=None):
    marker_dir = Path(checkpoints.region_count_track_names.get().output[0])
    return sorted(path.stem for path in marker_dir.glob("*.track"))


rule region_count_track:
    """Count one track's BAM events over ``regions.inputs.regions`` (CPU-only)."""
    input:
        tracks=str(TRACKS_DIR / "tracks.parquet"),
        marker=lambda w: str(
            Path(checkpoints.region_count_track_names.get().output[0]) / f"{w.track}.track"
        ),
    params:
        regions=REGIONS["inputs"]["regions"],
        anchor_regions=REGIONS["inputs"]["anchor_regions"],
        background_regions=REGIONS["inputs"]["background_regions"],
        out_dir=str(REGION_COUNTS_DIR),
        chrom_sizes_arg=_region_counts_chrom_sizes_arg,
        target_width=REGIONS["counts"]["target_width"],
        threads=REGIONS["counts"]["threads"],
    output:
        str(REGION_COUNTS_DIR / "{track}.parquet"),
    threads: REGIONS["counts"]["threads"]
    resources:
        gpu=0,
    log:
        str(RESULTS / "logs" / "region_count_track_{track}.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {log:q})"
        regulonado counts bam {params.regions:q} \
            --tracks {input.tracks:q} \
            --anchor-regions {params.anchor_regions:q} \
            --background-regions {params.background_regions:q} \
            --out-dir {params.out_dir:q} \
            --track {wildcards.track:q} \
            {params.chrom_sizes_arg} \
            --target-width {params.target_width} \
            --threads {params.threads} \
            > {log:q} 2>&1
        """


rule region_counts_gather:
    """Assemble every counted track's per-track cache into one ``RegionCountData`` run."""
    input:
        tracks=str(TRACKS_DIR / "tracks.parquet"),
        per_track=lambda w: expand(
            str(REGION_COUNTS_DIR / "{track}.parquet"), track=_region_track_names(w)
        ),
    params:
        regions=REGIONS["inputs"]["regions"],
        out_dir=str(REGION_COUNTS_DIR),
        dataset_dir=str(REGION_DATASET_DIR),
        chrom_sizes_arg=_region_counts_chrom_sizes_arg,
        target_width=REGIONS["counts"]["target_width"],
        val_chroms=" ".join(
            f"--val-chroms {shlex.quote(c)}" for c in REGIONS["counts"].get("val_chroms") or []
        ),
        test_chroms=" ".join(
            f"--test-chroms {shlex.quote(c)}" for c in REGIONS["counts"].get("test_chroms") or []
        ),
    output:
        regions=str(REGION_DATASET_DIR / "regions.parquet"),
        counts=str(REGION_DATASET_DIR / "counts.parquet"),
        tracks=str(REGION_DATASET_DIR / "tracks.parquet"),
    log:
        str(RESULTS / "logs" / "region_counts_gather.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {log:q})"
        regulonado counts gather {params.regions:q} \
            --tracks {input.tracks:q} \
            --out-dir {params.out_dir:q} \
            --dataset-dir {params.dataset_dir:q} \
            {params.chrom_sizes_arg} \
            --target-width {params.target_width} \
            {params.val_chroms} \
            {params.test_chroms} \
            > {log:q} 2>&1
        """


rule region_embed_chrom:
    """Cache one embedding's frozen features for one chromosome (GPU, resumable)."""
    input:
        regions=str(REGION_DATASET_DIR / "regions.parquet"),
        counts=str(REGION_DATASET_DIR / "counts.parquet"),
        tracks=str(REGION_DATASET_DIR / "tracks.parquet"),
        fasta=config["inputs"]["fasta"],
    params:
        dataset_dir=str(REGION_DATASET_DIR),
        backbone=lambda w: REGION_EMBEDDING_BY_NAME[w.embedding]["backbone"],
        out_dir=lambda w: str(_region_embed_out_dir(w.embedding)),
        extra=lambda w: _region_embed_command_flags(REGION_EMBEDDING_BY_NAME[w.embedding]),
    output:
        str(REGION_EMBEDDINGS_DIR / "{embedding}" / "{chrom}.parquet"),
    resources:
        # One GPU per embedding job, following the train_phase convention.
        gpu=1,
    wildcard_constraints:
        embedding="|".join(re.escape(name) for name in REGION_EMBEDDING_NAMES),
        chrom="|".join(re.escape(chrom) for chrom in REGION_CHROMS),
    log:
        str(RESULTS / "logs" / "region_embed_{embedding}_{chrom}.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {log:q})"
        regulonado embed regions {params.dataset_dir:q} {input.fasta:q} \
            --backbone {params.backbone:q} \
            --out {params.out_dir:q} \
            --chroms {wildcards.chrom:q} \
            {params.extra} \
            > {log:q} 2>&1
        """


rule region_embed_done:
    """Manifest sentinel: every chromosome cached for one embedding.

    ``manifest.parquet`` isn't declared as an input: it is written (and, on later
    calls, validated) as a side effect of *every* ``region_embed_chrom`` job for
    this embedding (see ``embed_regions``'s docstring), so it doesn't take
    ``{chrom}`` in its own name and can't be a per-chrom rule's tracked output
    without ambiguity. By the time every chrom file below exists, at least one
    such job has run, so the manifest is already on disk; this rule just checks it.
    """
    input:
        chroms=lambda w: expand(
            str(REGION_EMBEDDINGS_DIR / "{embedding}" / "{chrom}.parquet"),
            embedding=w.embedding,
            chrom=REGION_CHROMS,
        ),
    output:
        touch(str(REGION_EMBEDDINGS_DIR / "{embedding}" / ".done")),
    params:
        manifest=lambda w: str(REGION_EMBEDDINGS_DIR / w.embedding / "manifest.parquet"),
    wildcard_constraints:
        embedding="|".join(re.escape(name) for name in REGION_EMBEDDING_NAMES),
    run:
        manifest_path = Path(params.manifest)
        if not manifest_path.exists():
            raise WorkflowError(
                f"region_embed_done: expected {manifest_path} to exist once every "
                "chromosome had been embedded"
            )


def region_train_embed_done(wildcards):
    run = REGION_RUN_BY_NAME[wildcards.run]
    return str(REGION_EMBEDDINGS_DIR / run["embedding"] / ".done")


if REGION_TRAIN:

    rule region_train_phase:
        """One warm-started region-count-head training phase, chained like ``train_phase``."""
        input:
            embeddings=region_train_embed_done,
            previous=region_previous_phase_state,
        params:
            preset=lambda w: REGION_PHASE_BY_NAME[w.phase]["preset"],
            dataset_dir=str(REGION_DATASET_DIR),
            embeddings_dir=lambda w: str(
                _region_embed_out_dir(REGION_RUN_BY_NAME[w.run]["embedding"])
            ),
            target_group_arg=lambda w: (
                f"--target-group {shlex.quote(str(REGION_RUN_BY_NAME[w.run]['target_group']))}"
                if REGION_RUN_BY_NAME[w.run].get("target_group")
                else ""
            ),
            run_dir=lambda w: str(region_phase_run_dir(w.run, w.phase)),
            nproc=REGION_TRAIN["nproc_per_node"],
            overrides=_region_override_flags,
            prev_run_dir=lambda w: (
                ""
                if region_phase_index(w.phase) == 0
                else str(
                    region_phase_run_dir(w.run, REGION_PHASE_NAMES[region_phase_index(w.phase) - 1])
                )
            ),
            resolver=str(Path(workflow.basedir) / "scripts" / "resolve_checkpoint.py"),
        output:
            state=str(REGION_TRAIN_DIR / "{run}" / "{phase}" / "trainer_state.json"),
        resources:
            gpu=REGION_TRAIN["nproc_per_node"],
        wildcard_constraints:
            run="|".join(re.escape(name) for name in REGION_RUN_NAMES),
            phase="|".join(re.escape(name) for name in REGION_PHASE_NAMES),
        log:
            str(RESULTS / "logs" / "region_train_{run}_{phase}.log"),
        shell:
            r"""
            set -euo pipefail

            INIT_ARGS=()
            if [ -n "{params.prev_run_dir}" ]; then
                CKPT=$(python {params.resolver:q} "{params.prev_run_dir}")
                echo "Warm-starting regions/{wildcards.run}/{wildcards.phase} from: $CKPT"
                INIT_ARGS=(--init-weights-from-checkpoint "$CKPT")
            fi

            regulonado train-regions \
                {params.dataset_dir:q} \
                --embeddings {params.embeddings_dir:q} \
                --preset {params.preset:q} \
                {params.target_group_arg} \
                --output-dir {params.run_dir:q} \
                --nproc-per-node {params.nproc} \
                {params.overrides} \
                "${{INIT_ARGS[@]}}" \
                > {log:q} 2>&1
            """
