"""Scale-factor inference and metadata enrichment."""


rule scale_factors:
    """Infer per-track scale factors so tracks are comparable in magnitude.

    One unit of work: read the built dataset's track statistics and emit one
    scale factor per track.

    All methods first infer original library sizes with BamNado. ``tmm`` then
    adds a dataset-derived trimmed-mean correction; ``bamnado`` instead
    corrects using one of BamNado's own BAM-based methods (tmm,
    csaw-background, cpm, median-of-ratios, spike-in); ``seqnado`` reuses the
    spike-in factors an upstream SeqNado run already computed.

    ``seqnado`` applies to a single project only — its factors are not
    comparable between projects, so aggregated datasets must use ``tmm``.
    """
    input:
        metadata=str(DATASET_DIR / "regulonado_metadata.json"),
    params:
        method=config["scaling"]["method"],
        initial=str(SCALING_DIR / "initial_scale_factors.parquet"),
        bam_dir=config["inputs"].get("bam_dir", ""),
        bamnado_method=config["scaling"].get("bamnado_method", "csaw-background"),
        bamnado_exogenous_prefix=config["scaling"].get("bamnado_exogenous_prefix", ""),
        seqnado_project=(
            seqnado_scaling_project() if config["scaling"]["method"] == "seqnado" else ""
        ),
        seqnado_spikein_method=config["scaling"].get("seqnado_spikein_method", ""),
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
        elif [ "{params.method}" = "seqnado" ]; then
            regulonado normalization original {input.metadata:q} --output {params.initial:q}
            if [ -n "{params.seqnado_spikein_method}" ]; then
                regulonado normalization seqnado {input.metadata:q} \
                    --project {params.seqnado_project:q} \
                    --method {params.seqnado_spikein_method:q} \
                    --scale-factors {params.initial:q} \
                    --output {output.parquet:q}
            else
                regulonado normalization seqnado {input.metadata:q} \
                    --project {params.seqnado_project:q} \
                    --scale-factors {params.initial:q} \
                    --output {output.parquet:q}
            fi
        elif [ "{params.method}" = "bamnado" ]; then
            regulonado normalization original {input.metadata:q} --output {params.initial:q}
            if [ -n "{params.bamnado_exogenous_prefix}" ]; then
                regulonado normalization bamnado {input.metadata:q} \
                    --bam-dir {params.bam_dir:q} \
                    --method {params.bamnado_method:q} \
                    --exogenous-prefix {params.bamnado_exogenous_prefix:q} \
                    --scale-factors {params.initial:q} \
                    --output {output.parquet:q}
            else
                regulonado normalization bamnado {input.metadata:q} \
                    --bam-dir {params.bam_dir:q} \
                    --method {params.bamnado_method:q} \
                    --scale-factors {params.initial:q} \
                    --output {output.parquet:q}
            fi
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
