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
        bigwig_dir=config["inputs"]["bigwig_dir"],
        initial=str(SCALING_DIR / "initial_scale_factors.parquet"),
        subcommand=lambda w: (
            "normalization tmm"
            if config["scaling"]["method"] == "tmm"
            else "normalization original"
        ),
    output:
        parquet=str(SCALING_DIR / "scale_factors.parquet"),
    log:
        str(RESULTS / "logs" / "scale_factors.log"),
    shell:
        r"""
        if [ "{params.method}" = "tmm" ]; then
            regulonado normalization infer {params.bigwig_dir} --output {params.initial}
            regulonado normalization tmm {input.metadata} {params.initial} --output {output.parquet}
        else
            regulonado normalization original {input.metadata} --output {output.parquet}
        fi > {log} 2>&1
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
