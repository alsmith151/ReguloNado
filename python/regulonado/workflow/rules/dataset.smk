"""Dataset construction rules: BigWig + FASTA -> Arrow DatasetDict."""


rule build_dataset:
    """Build the Arrow dataset from BigWig tracks and a reference FASTA.

    One unit of work: scan every BigWig once per chromosome, bin the signal, join
    it to one-hot encoded sequence, and write one Arrow IPC shard per chromosome
    per split. This is the expensive step — it is I/O-bound on BigWig decoding,
    which is why `stage_to_scratch` matters on networked storage.

    Outputs the dataset directory plus the metadata file that the scaling rules
    consume. `dataset_info.json` is used as the completion sentinel because the
    builder writes it last.
    """
    input:
        intervals=config["inputs"]["intervals"],
        fasta=config["inputs"]["fasta"],
    params:
        bigwig_dir=config["inputs"]["bigwig_dir"],
        out_dir=lambda w, output: str(Path(output.info).parent),
        context_length=config["build"]["context_length"],
        bin_size=config["build"]["bin_size"],
        n_pred_bins=config["build"]["n_pred_bins"],
        shift_max_bp=config["build"]["shift_max_bp"],
        extract_threads=config["build"]["extract_threads"],
        arrow_write_threads=config["build"]["arrow_write_threads"],
        arrow_batch_size=config["build"]["arrow_batch_size"],
        compression=config["build"]["compression"],
        stage=lambda w: "--stage" if config["build"]["stage_to_scratch"] else "--no-stage",
        drop_missing=lambda w: "--drop-missing" if config["build"]["drop_missing"] else "",
        dedupe=config["build"]["dedupe_tracks"],
    output:
        info=str(DATASET_DIR / "dataset_dict.json"),
        metadata=str(DATASET_DIR / "regulonado_metadata.json"),
    threads: config["build"]["extract_threads"]
    log:
        str(RESULTS / "logs" / "build_dataset.log"),
    shell:
        r"""
        regulonado build \
            {input.intervals} \
            {input.fasta} \
            {params.out_dir} \
            --bigwig-dir {params.bigwig_dir} \
            --context-length {params.context_length} \
            --bin-size {params.bin_size} \
            --n-pred-bins {params.n_pred_bins} \
            --shift-max-bp {params.shift_max_bp} \
            --n-extract-threads {params.extract_threads} \
            --arrow-write-threads {params.arrow_write_threads} \
            --arrow-batch-size {params.arrow_batch_size} \
            --arrow-compression {params.compression} \
            --dedupe-tracks {params.dedupe} \
            {params.stage} {params.drop_missing} \
            > {log} 2>&1
        """


rule recompress_dataset:
    """Rebatch and recompress the dataset for faster random-access reads.

    One unit of work: rewrite every Arrow shard with smaller record batches and
    zstd compression. Training reads randomly, so smaller batches reduce the
    amount decoded per sample.

    Skipped entirely when `recompress.enabled` is false — in that case the
    training rules read the raw build output instead.
    """
    input:
        info=str(DATASET_DIR / "dataset_dict.json"),
    params:
        src=str(DATASET_DIR),
        dst=str(RECHUNK_DIR),
        zstd_level=config["recompress"]["zstd_level"],
        max_batch_size=config["recompress"]["max_batch_size"],
        workers=config["recompress"]["workers"],
    output:
        info=str(RECHUNK_DIR / "dataset_dict.json"),
    threads: config["recompress"]["workers"]
    log:
        str(RESULTS / "logs" / "recompress_dataset.log"),
    shell:
        r"""
        regulonado recompress-dataset \
            {params.src} \
            {params.dst} \
            --level {params.zstd_level} \
            --max-batch-size {params.max_batch_size} \
            --workers {params.workers} \
            > {log} 2>&1
        """
