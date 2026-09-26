"""Trunk embedding caches for ``trunk: cached`` runs, one per distinct trunk setup.

Caches are derived from the runs, not declared: runs whose backbone, checkpoint and
cache settings match share one cache (``TrainRun.cache_name``). Each cache is built one
chromosome per GPU job over region_counts.smk's ``region_set``, so embedding runs while
the BAMs are still being counted.
"""

import re
import shlex
from pathlib import Path

from snakemake.exceptions import WorkflowError

from regulonado.normalization import read_regions

EMBEDDINGS_DIR = RESULTS / "embeddings"
# cache name -> the run (any one) whose trunk setup defines it.
EMBEDDING_CACHES = {
    run.cache_name(): run for run in _VALIDATED.train.runs if run.trunk == "cached"
}


def embedding_cache_dir(run_name):
    return EMBEDDINGS_DIR / _VALIDATED.train.run(run_name).cache_name()


def _region_chroms():
    """Distinct chromosomes of the configured region set, read once at parse time."""
    seen: dict[str, None] = {}
    for chrom, _start, _end in read_regions(Path(REGION_COUNTS["regions"])):
        seen.setdefault(chrom, None)
    return sorted(seen)


EMBED_CHROMS = _region_chroms()


def _embed_flags(cache_name):
    run = EMBEDDING_CACHES[cache_name]
    cache = run.cache
    flags = ["--backbone", run.backbone.type, "--pretrained", shlex.quote(run.backbone.pretrained)]
    if cache is not None:
        if cache.rc:
            flags.append("--rc")
        if cache.pool_to:
            flags += ["--pool-to", str(cache.pool_to)]
        if cache.context:
            flags += ["--context", str(cache.context)]
        if cache.stride:
            flags += ["--stride", str(cache.stride)]
        flags += ["--batch-size", str(cache.batch_size)]
    return " ".join(flags)


rule embed_chrom:
    """Cache one trunk's embeddings for one chromosome (GPU, resumable)."""
    input:
        regions=str(REGION_SET),
        fasta=config["inputs"]["fasta"],
    params:
        out_dir=lambda w: str(EMBEDDINGS_DIR / w.cache),
        flags=lambda w: _embed_flags(w.cache),
    output:
        str(EMBEDDINGS_DIR / "{cache}" / "{chrom}.parquet"),
    resources:
        gpu=1,
    wildcard_constraints:
        cache="|".join(re.escape(name) for name in EMBEDDING_CACHES),
        chrom="|".join(re.escape(chrom) for chrom in EMBED_CHROMS),
    log:
        str(RESULTS / "logs" / "embed_{cache}_{chrom}.log"),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname {log:q})"
        regulonado embed regions {input.regions:q} {input.fasta:q} \
            --out {params.out_dir:q} \
            --chroms {wildcards.chrom:q} \
            {params.flags} \
            > {log:q} 2>&1
        """


rule embed_done:
    """Manifest sentinel: every chromosome cached.

    ``manifest.parquet`` is written (and, on later calls, validated) as a side effect of
    every ``embed_chrom`` job for this cache, so it can't be one per-chrom rule's tracked
    output; once every chromosome file exists it is on disk, and this rule checks it.
    """
    input:
        chroms=lambda w: expand(
            str(EMBEDDINGS_DIR / "{cache}" / "{chrom}.parquet"), cache=w.cache, chrom=EMBED_CHROMS
        ),
    output:
        touch(str(EMBEDDINGS_DIR / "{cache}" / ".done")),
    params:
        manifest=lambda w: str(EMBEDDINGS_DIR / w.cache / "manifest.parquet"),
    wildcard_constraints:
        cache="|".join(re.escape(name) for name in EMBEDDING_CACHES),
    run:
        if not Path(params.manifest).exists():
            raise WorkflowError(
                f"embed_done: expected {params.manifest} once every chromosome was embedded"
            )
