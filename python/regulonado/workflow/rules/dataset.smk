"""Dataset construction: FASTA + tracks.parquet -> Arrow DatasetDict."""

from regulonado.dataset.build import DEFAULT_SPLITS

SPLITS = list(DEFAULT_SPLITS)  # train / validation / test
maybe_temp = temp if config["recompress"]["enabled"] else (lambda x: x)


rule build_dataset:
    """Build the Arrow dataset from a reference FASTA and the assembled track table.

    One unit of work: scan every BigWig once per chromosome, bin the signal, join
    it to one-hot encoded sequence, and write one Arrow IPC shard per chromosome
    per split. This is the expensive step — it is I/O-bound on BigWig decoding,
    which is why `stage_to_scratch` matters on networked storage.

    Track identity, dedupe and QC are already settled by `track_assemble`; this
    rule only reads `tracks.parquet` and writes it straight back out alongside
    the Arrow shards (`dataset/tracks.parquet`), with build-time scalars merged
    in. `dataset_info.json` is used as the completion sentinel because the
    builder writes it last.

    With `recompress.enabled`, the raw build output here is a pure
    intermediate — `temp()` marks it (and its split directories) for cleanup
    once `recompress_dataset` has consumed it.
    """
    input:
        intervals=config["inputs"]["intervals"],
        fasta=config["inputs"]["fasta"],
        table=str(TRACKS_DIR / "tracks.parquet"),
    params:
        out_dir=lambda w, output: str(Path(output.info).parent),
        context_length=config["dataset"]["context_length"],
        bin_size=config["dataset"]["bin_size"],
        n_pred_bins=config["dataset"]["n_pred_bins"],
        shift_max_bp=config["dataset"]["shift_max_bp"],
        extract_threads=config["dataset"]["extract_threads"],
        arrow_write_threads=config["dataset"]["arrow_write_threads"],
        arrow_batch_size=config["dataset"]["arrow_batch_size"],
        compression=config["dataset"]["compression"],
        stage=lambda w: "--stage" if config["dataset"]["stage_to_scratch"] else "--no-stage",
    output:
        info=maybe_temp(str(DATASET_DIR / "dataset_dict.json")),
        table=maybe_temp(str(DATASET_DIR / "tracks.parquet")),
        splits=[maybe_temp(directory(str(DATASET_DIR / s))) for s in SPLITS],
    threads: config["dataset"]["extract_threads"]
    log:
        str(RESULTS / "logs" / "build_dataset.log"),
    shell:
        r"""
        regulonado dataset \
            {input.intervals} \
            {input.fasta} \
            {params.out_dir} \
            --track-table {input.table:q} \
            --context-length {params.context_length} \
            --bin-size {params.bin_size} \
            --n-pred-bins {params.n_pred_bins} \
            --shift-max-bp {params.shift_max_bp} \
            --n-extract-threads {params.extract_threads} \
            --arrow-write-threads {params.arrow_write_threads} \
            --arrow-batch-size {params.arrow_batch_size} \
            --arrow-compression {params.compression} \
            {params.stage} \
            > {log} 2>&1
        """


rule recompress_dataset:
    """Rebatch and recompress the dataset for faster random-access reads.

    One unit of work: rewrite every Arrow shard with smaller record batches and
    zstd compression. Training reads randomly, so smaller batches reduce the
    amount decoded per sample.

    Skipped entirely when `recompress.enabled` is false — in that case the
    training rules read the raw build output instead, and `build_dataset`'s
    outputs are ordinary (non-temp) files.
    """
    input:
        info=str(DATASET_DIR / "dataset_dict.json"),
        table=str(DATASET_DIR / "tracks.parquet"),
    params:
        src=str(DATASET_DIR),
        dst=str(RECHUNK_DIR),
        zstd_level=config["recompress"]["zstd_level"],
        max_batch_size=config["recompress"]["max_batch_size"],
        workers=config["recompress"]["workers"],
    output:
        info=str(RECHUNK_DIR / "dataset_dict.json"),
        table=str(RECHUNK_DIR / "tracks.parquet"),
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
            --overwrite \
            > {log} 2>&1
        """
