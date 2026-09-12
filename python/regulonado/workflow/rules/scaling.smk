"""Scale-factor inference, run directly from BigWigs before the Arrow build."""

import json


rule scale_factors:
    """Infer per-track scale factors so tracks are comparable in magnitude.

    One unit of work: read the discovered tracks' BigWig statistics and emit
    one scale factor per track, keyed by ``track_name``.

    All methods first infer original library sizes with BamNado. ``tmm`` then
    adds a dataset-derived trimmed-mean correction, scanning ``inputs.intervals``
    directly (sharing the scan with interval-based QC checks via
    ``track_interval_means``); ``bamnado`` instead corrects using one of
    BamNado's own BAM-based methods (tmm, csaw-background, cpm,
    median-of-ratios, spike-in); ``seqnado`` reuses the spike-in factors an
    upstream SeqNado run already computed.

    ``seqnado`` applies to a single project only — its factors are not
    comparable between projects, so aggregated datasets must use ``tmm``.
    """
    input:
        table=str(TRACKS_STAGE_DIR / "discovered.parquet"),
        intervals=(
            config["inputs"]["intervals"] if config["scaling"]["method"] == "tmm" else []
        ),
        interval_means=(
            str(TRACKS_STAGE_DIR / "interval_means.parquet")
            if config["scaling"]["method"] == "tmm"
            else []
        ),
    params:
        method=config["scaling"]["method"],
        initial=str(TRACKS_STAGE_DIR / "initial_scale_factors.parquet"),
        bin_size=config["dataset"]["bin_size"],
        n_pred_bins=config["dataset"]["n_pred_bins"],
        shift_max_bp=config["dataset"]["shift_max_bp"],
        bam_dir=config["inputs"].get("bam_dir", ""),
        bamnado_method=config["scaling"].get("bamnado_method", "csaw-background"),
        bamnado_exogenous_prefix=config["scaling"].get("bamnado_exogenous_prefix", ""),
        seqnado_project=(
            seqnado_scaling_project() if config["scaling"]["method"] == "seqnado" else ""
        ),
        seqnado_spikein_method=config["scaling"].get("seqnado_spikein_method", ""),
        anchor_regions=config["scaling"].get("anchor_regions", ""),
        background_regions=config["scaling"].get("background_regions", ""),
        heldout_regions=config["scaling"].get("heldout_regions", ""),
        heldout_arg=(
            "--heldout-regions " + config["scaling"]["heldout_regions"]
            if config["scaling"].get("heldout_regions") else ""
        ),
        window_stat_bp=config["scaling"].get("window_stat_bp", 1000),
        window_stat_bp_by_assay_arg=(
            "--window-stat-bp-by-assay "
            + shlex.quote(json.dumps(config["scaling"]["window_stat_bp_by_assay"]))
            if config["scaling"].get("window_stat_bp_by_assay") else ""
        ),
        background_sample_arg=(
            "--background-sample " + str(config["scaling"]["background_sample"])
            if config["scaling"].get("background_sample") is not None else ""
        ),
    output:
        parquet=str(TRACKS_STAGE_DIR / "scale_factors.parquet"),
    log:
        str(RESULTS / "logs" / "scale_factors.log"),
    shell:
        r"""
        if [ "{params.method}" = "tmm" ]; then
            regulonado normalization original {input.table:q} --bin-size {params.bin_size} \
                --output {params.initial:q}
            regulonado normalization tmm {input.table:q} \
                --interval-means {input.interval_means:q} \
                --scale-factors {params.initial:q} \
                --bin-size {params.bin_size} \
                --n-pred-bins {params.n_pred_bins} \
                --shift-max-bp {params.shift_max_bp} \
                --output {output.parquet:q}
        elif [ "{params.method}" = "seqnado" ]; then
            regulonado normalization original {input.table:q} --bin-size {params.bin_size} \
                --output {params.initial:q}
            if [ -n "{params.seqnado_spikein_method}" ]; then
                regulonado normalization seqnado {input.table:q} \
                    --project {params.seqnado_project:q} \
                    --method {params.seqnado_spikein_method:q} \
                    --scale-factors {params.initial:q} \
                    --output {output.parquet:q}
            else
                regulonado normalization seqnado {input.table:q} \
                    --project {params.seqnado_project:q} \
                    --scale-factors {params.initial:q} \
                    --output {output.parquet:q}
            fi
        elif [ "{params.method}" = "bamnado" ]; then
            regulonado normalization original {input.table:q} --bin-size {params.bin_size} \
                --output {params.initial:q}
            if [ -n "{params.bamnado_exogenous_prefix}" ]; then
                regulonado normalization bamnado {input.table:q} \
                    --bam-dir {params.bam_dir:q} \
                    --method {params.bamnado_method:q} \
                    --exogenous-prefix {params.bamnado_exogenous_prefix:q} \
                    --scale-factors {params.initial:q} \
                    --output {output.parquet:q}
            else
                regulonado normalization bamnado {input.table:q} \
                    --bam-dir {params.bam_dir:q} \
                    --method {params.bamnado_method:q} \
                    --scale-factors {params.initial:q} \
                    --output {output.parquet:q}
            fi
        elif [ "{params.method}" = "anchor" ]; then
            regulonado normalization anchor {input.table:q} \
                --anchor-regions {params.anchor_regions:q} \
                --background-regions {params.background_regions:q} \
                {params.heldout_arg} \
                --bin-size {params.bin_size} \
                --window-stat-bp {params.window_stat_bp} \
                {params.window_stat_bp_by_assay_arg} \
                {params.background_sample_arg} \
                --output {output.parquet:q}
        else
            regulonado normalization original {input.table:q} --bin-size {params.bin_size} \
                --output {output.parquet:q}
        fi > {log} 2>&1
        """
