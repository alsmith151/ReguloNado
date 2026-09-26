"""The region-count dataset (``targets.region_counts``): one count per region per track.

Included only when ``targets.region_counts`` is configured::

    region_set -> region_counts/region_set.parquet   (canonical rows, CPU, no BAMs)
      |-- region_count_track (per track, CPU) -> region_counts_gather -> region_counts/dataset
      `-- embeddings.smk's cache jobs read the same rows, so they run while BAMs are counted

Tracks are the shared ``tracks.parquet``: discovered from BAMs, or bigWig tracks that
carry their BAM. The track list isn't known at DAG-construction time, so
``region_count_track_names`` is a ``checkpoint`` -- the same pattern
``rules/design.smk``'s ``shard_candidates`` uses.
"""

import shlex
from pathlib import Path

REGION_COUNTS_DIR = RESULTS / "region_counts"
REGION_SET = REGION_COUNTS_DIR / "region_set.parquet"
REGION_TRACK_COUNTS_DIR = REGION_COUNTS_DIR / "counts" / "tracks"
REGION_DATASET_DIR = REGION_COUNTS_DIR / "dataset"
# One anchor/background definition serves both track formats unless overridden here.
REGION_ANCHORS = REGION_COUNTS.get("anchor_regions") or config["scaling"].get("anchor_regions")
REGION_BACKGROUND = REGION_COUNTS.get("background_regions") or config["scaling"].get(
    "background_regions"
)


def _chrom_sizes_arg(wildcards):
    chrom_sizes = REGION_COUNTS.get("chrom_sizes")
    return f"--chrom-sizes {shlex.quote(str(chrom_sizes))}" if chrom_sizes else ""


rule region_set:
    """Canonical region table shared by counting and embedding (no BAMs read)."""
    input:
        regions=REGION_COUNTS["regions"],
    params:
        chrom_sizes_arg=_chrom_sizes_arg,
        target_width=REGION_COUNTS["target_width"],
        val_chroms=" ".join(
            f"--val-chroms {shlex.quote(c)}" for c in REGION_COUNTS.get("val_chroms") or []
        ),
        test_chroms=" ".join(
            f"--test-chroms {shlex.quote(c)}" for c in REGION_COUNTS.get("test_chroms") or []
        ),
    output:
        str(REGION_SET),
    resources:
        gpu=0,
    log:
        str(RESULTS / "logs" / "region_set.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {log:q})"
        regulonado counts regions {input.regions:q} \
            --output {output:q} \
            {params.chrom_sizes_arg} \
            --target-width {params.target_width} \
            {params.val_chroms} \
            {params.test_chroms} \
            > {log:q} 2>&1
        """


checkpoint region_count_track_names:
    """Snapshot the count-track names into one marker file per track.

    Reading the table here also fails fast on a track with no BAM or no known assay.
    """
    input:
        table=str(TRACKS_DIR / "tracks.parquet"),
    output:
        directory(str(REGION_COUNTS_DIR / "counts" / "_track_names")),
    run:
        from regulonado.counts.bam import read_count_tracks

        out_dir = Path(output[0])
        out_dir.mkdir(parents=True, exist_ok=True)
        for track_name in read_count_tracks(input.table)["track_name"]:
            (out_dir / f"{track_name}.track").touch()


def _region_track_names(wildcards=None):
    marker_dir = Path(checkpoints.region_count_track_names.get().output[0])
    return sorted(path.stem for path in marker_dir.glob("*.track"))


rule region_count_track:
    """Count one track's BAM events over the canonical region set (CPU-only)."""
    input:
        regions=str(REGION_SET),
        tracks=str(TRACKS_DIR / "tracks.parquet"),
        anchor_regions=REGION_ANCHORS,
        background_regions=REGION_BACKGROUND,
        marker=lambda w: str(
            Path(checkpoints.region_count_track_names.get().output[0]) / f"{w.track}.track"
        ),
    params:
        out_dir=str(REGION_TRACK_COUNTS_DIR),
        target_width=REGION_COUNTS["target_width"],
        threads=REGION_COUNTS["threads"],
    output:
        str(REGION_TRACK_COUNTS_DIR / "{track}.parquet"),
    threads: REGION_COUNTS["threads"]
    resources:
        gpu=0,
    log:
        str(RESULTS / "logs" / "region_count_track_{track}.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {log:q})"
        regulonado counts bam {input.regions:q} \
            --tracks {input.tracks:q} \
            --anchor-regions {input.anchor_regions:q} \
            --background-regions {input.background_regions:q} \
            --out-dir {params.out_dir:q} \
            --track {wildcards.track:q} \
            --target-width {params.target_width} \
            --threads {params.threads} \
            > {log:q} 2>&1
        """


rule region_counts_gather:
    """Assemble every counted track's per-track cache into one ``RegionCountData`` run."""
    input:
        regions=str(REGION_SET),
        tracks=str(TRACKS_DIR / "tracks.parquet"),
        per_track=lambda w: expand(
            str(REGION_TRACK_COUNTS_DIR / "{track}.parquet"), track=_region_track_names(w)
        ),
    params:
        out_dir=str(REGION_TRACK_COUNTS_DIR),
        dataset_dir=str(REGION_DATASET_DIR),
        target_width=REGION_COUNTS["target_width"],
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
        regulonado counts gather {input.regions:q} \
            --tracks {input.tracks:q} \
            --out-dir {params.out_dir:q} \
            --dataset-dir {params.dataset_dir:q} \
            --target-width {params.target_width} \
            > {log:q} 2>&1
        """
