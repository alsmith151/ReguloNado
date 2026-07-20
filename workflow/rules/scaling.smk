"""Scale-factor inference and metadata enrichment."""


rule scale_factors:
    """Infer per-track scale factors so tracks are comparable in magnitude.

    One unit of work: read the built dataset's track statistics and emit one
    scale factor per track.

    Two methods, selected by `scaling.method`:

    * ``tmm``      — trimmed mean of M-values, computed from the dataset itself.
                     No external tools needed. This is the default.
    * ``original`` — derived from the original BigWig library sizes. Requires the
                     BamNado binary on PATH; it is not pip-installable, so this
                     method will fail on a plain `pip install regulonado`.
    """
    input:
        metadata=str(DATASET_DIR / "regulonado_metadata.json"),
    params:
        method=config["scaling"]["method"],
        subcommand=lambda w: (
            "calculate-tmm-scaling"
            if config["scaling"]["method"] == "tmm"
            else "calculate-original-scaling"
        ),
    output:
        parquet=str(SCALING_DIR / "scale_factors.parquet"),
    log:
        str(RESULTS / "logs" / "scale_factors.log"),
    shell:
        r"""
        regulonado {params.subcommand} \
            {input.metadata} \
            --output {output.parquet} \
            > {log} 2>&1
        """


rule enrich_metadata:
    """Fold the scale factors into the dataset metadata.

    One unit of work: merge scale_factors.parquet into a copy of the dataset
    metadata, producing the file the training phases read. Writing to a new path
    rather than mutating the builder's output keeps the build rule's outputs
    immutable, so re-running enrichment never invalidates the expensive build.
    """
    input:
        metadata=str(DATASET_DIR / "regulonado_metadata.json"),
        parquet=str(SCALING_DIR / "scale_factors.parquet"),
    output:
        enriched=str(ENRICHED_METADATA),
    log:
        str(RESULTS / "logs" / "enrich_metadata.log"),
    shell:
        r"""
        cp {input.metadata} {output.enriched}
        regulonado enrich-metadata {output.enriched} {input.parquet} > {log} 2>&1
        """
