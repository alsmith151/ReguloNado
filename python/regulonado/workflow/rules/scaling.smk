"""Scale-factor inference and metadata enrichment."""


rule scale_factors:
    """Infer per-track scale factors so tracks are comparable in magnitude.

    One unit of work: read the built dataset's track statistics and emit one
    scale factor per track.

    Both methods first infer original library sizes with BamNado. ``tmm`` then
    adds a dataset-derived trimmed-mean correction.
    """
    input:
        metadata=str(DATASET_DIR / "regulonado_metadata.json"),
    params:
        method=config["scaling"]["method"],
        initial=str(SCALING_DIR / "initial_scale_factors.parquet"),
    output:
        parquet=str(SCALING_DIR / "scale_factors.parquet"),
    log:
        str(RESULTS / "logs" / "scale_factors.log"),
    shell:
        r"""
        if [ "{params.method}" = "tmm" ]; then
            regulonado normalization original {input.metadata:q} --output {params.initial:q}
            regulonado normalization tmm {input.metadata:q} \
                --scale-factors {params.initial:q} \
                --output {output.parquet:q}
        else
            regulonado normalization original {input.metadata:q} --output {output.parquet:q}
        fi > {log} 2>&1
        """


rule enrich_metadata:
    """Write a scale-factor-enriched copy of the dataset metadata."""
    input:
        metadata=str(DATASET_DIR / "regulonado_metadata.json"),
        parquet=str(SCALING_DIR / "scale_factors.parquet"),
    output:
        enriched=str(ENRICHED_METADATA),
    log:
        str(RESULTS / "logs" / "enrich_metadata.log"),
    shell:
        r"""
        regulonado enrich-metadata \
            {input.metadata:q} \
            {input.parquet:q} \
            --output {output.enriched:q} \
            > {log:q} 2>&1
        """
