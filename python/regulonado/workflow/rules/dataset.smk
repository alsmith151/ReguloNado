"""The profile dataset: FASTA + bigWig tracks.parquet -> Parquet shards (``targets.profile``)."""

from regulonado.dataset.build import DEFAULT_SPLITS

SPLITS = list(DEFAULT_SPLITS)  # train / validation / test


rule build_dataset:
    """Build the Parquet dataset from a reference FASTA and the assembled track table.

    One unit of work: scan every BigWig once per chromosome, bin the signal, join
    it to sequence tokens, and write Parquet shards per split. This is the expensive
    step — it is I/O-bound on BigWig decoding, which is why `stage_to_scratch` matters
    on networked storage.

    Track identity, dedupe and QC are already settled by `track_assemble`; this
    rule only reads `tracks.parquet` and writes it straight back out alongside
    the Parquet shards (`dataset/tracks.parquet`), with build-time scalars merged
    in. `dataset/README.md` is written last and serves as the completion sentinel.
    """
    input:
        intervals=PROFILE["intervals"],
        fasta=config["inputs"]["fasta"],
        table=str(TRACKS_DIR / "tracks.parquet"),
    params:
        out_dir=lambda w, output: str(Path(output.readme).parent),
        context_length=PROFILE["context_length"],
        bin_size=PROFILE["bin_size"],
        n_pred_bins=PROFILE["n_pred_bins"],
        shift_max_bp=PROFILE["shift_max_bp"],
        extract_threads=PROFILE["extract_threads"],
        write_threads=PROFILE["write_threads"],
        zstd_level=PROFILE["zstd_level"],
        rows_per_row_group=PROFILE["rows_per_row_group"],
        bin_denominator=PROFILE["bin_denominator"],
        missing_bins=PROFILE["missing_bins"],
        stage=lambda w: "--stage" if PROFILE["stage_to_scratch"] else "--no-stage",
    output:
        readme=str(DATASET_DIR / "README.md"),
        table=str(DATASET_DIR / "tracks.parquet"),
        data=directory(str(DATASET_DIR / "data")),
    threads: PROFILE["extract_threads"]
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
            --write-threads {params.write_threads} \
            --zstd-level {params.zstd_level} \
            --rows-per-row-group {params.rows_per_row_group} \
            --bin-denominator {params.bin_denominator} \
            --missing-bins {params.missing_bins} \
            {params.stage} \
            > {log} 2>&1
        """
