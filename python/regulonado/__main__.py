from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(
    ctx: typer.Context,
    dataset: Annotated[
        Path,
        typer.Argument(help="Saved Regulonado/Hugging Face dataset directory"),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Run directory for checkpoints and diagnostics"),
    ] = None,
    experiment: Annotated[
        str,
        typer.Option("--experiment", "-e", help="Hydra experiment config to launch"),
    ] = "condition_agnostic_borzoi",
    nproc_per_node: Annotated[
        int,
        typer.Option(
            "--nproc-per-node",
            help="Use torchrun with this many local processes when >1",
        ),
    ] = 1,
    resume_from_checkpoint: Annotated[
        Optional[str],
        typer.Option(
            "--resume-from-checkpoint",
            help="Full Trainer resume from a checkpoint dir, or 'true' for latest in output-dir",
        ),
    ] = None,
    init_weights_from_checkpoint: Annotated[
        Optional[Path],
        typer.Option(
            "--init-weights-from-checkpoint",
            help="Warm start from model weights only with a fresh optimizer/scheduler",
        ),
    ] = None,
    max_steps: Annotated[
        Optional[int],
        typer.Option("--max-steps", help="Override trainer.max_steps"),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", help="Override per-device train batch size"),
    ] = None,
    eval_batch_size: Annotated[
        Optional[int],
        typer.Option("--eval-batch-size", help="Override per-device eval batch size"),
    ] = None,
    learning_rate: Annotated[
        Optional[float],
        typer.Option("--learning-rate", "--lr", help="Override head learning rate"),
    ] = None,
    backbone_lr: Annotated[
        Optional[float],
        typer.Option("--backbone-lr", help="Override backbone learning rate"),
    ] = None,
    num_workers: Annotated[
        Optional[int],
        typer.Option("--num-workers", help="Override DataLoader worker count"),
    ] = None,
    no_wandb: Annotated[
        bool,
        typer.Option("--no-wandb", help="Disable W&B reporting for this run"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the resolved command without running it"),
    ] = False,
) -> None:
    """Train a model with friendly options plus optional raw Hydra overrides.

    Extra arguments after the options are passed directly to Hydra, for example:

    regulonado train dataset/ --max-steps 1000 trainer.fit_examples.num_examples=8
    """
    if resume_from_checkpoint and init_weights_from_checkpoint:
        typer.echo(
            "Set only one of --resume-from-checkpoint or --init-weights-from-checkpoint.",
            err=True,
        )
        raise typer.Exit(1)

    overrides = [
        f"+experiment={experiment}",
        f"data.path={dataset}",
    ]
    if output_dir is not None:
        overrides.append(f"output_dir={output_dir}")
    if resume_from_checkpoint is not None:
        overrides.append(f"trainer.resume_from_checkpoint={resume_from_checkpoint}")
    if init_weights_from_checkpoint is not None:
        overrides.append(
            f"trainer.init_weights_from_checkpoint={init_weights_from_checkpoint}"
        )
    if max_steps is not None:
        overrides.append(f"trainer.max_steps={max_steps}")
    if batch_size is not None:
        overrides.append(f"trainer.batch_size={batch_size}")
    if eval_batch_size is not None:
        overrides.append(f"trainer.eval_batch_size={eval_batch_size}")
    if learning_rate is not None:
        overrides.append(f"trainer.learning_rate={learning_rate}")
    if backbone_lr is not None:
        overrides.append(f"trainer.backbone_learning_rate={backbone_lr}")
    if num_workers is not None:
        overrides.append(f"trainer.num_workers={num_workers}")
    if no_wandb:
        overrides.append("trainer.report_to=[]")
        overrides.append("trainer.fit_examples.log_to_wandb=false")
    overrides.extend(ctx.args)

    if nproc_per_node > 1:
        command = [
            "torchrun",
            f"--nproc_per_node={nproc_per_node}",
            "-m",
            "regulonado.train",
            *overrides,
        ]
    else:
        command = [sys.executable, "-m", "regulonado.train", *overrides]

    typer.echo(" ".join(command))
    if dry_run:
        return
    raise typer.Exit(subprocess.run(command).returncode)


@app.command()
def scale(
    bigwig_dir: Annotated[Path, typer.Argument(help="Directory containing .bw / .bigwig files")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")],
    fmt: Annotated[str, typer.Option("--format", "-f", help="Output format: csv or parquet")] = "parquet",
    max_workers: Annotated[int, typer.Option("--workers", "-w", help="Thread pool size")] = 16,
    glob: Annotated[str, typer.Option("--glob", help="Glob pattern for bigwig files")] = "*.bw",
) -> None:
    """Infer scale factors for all bigwig files in a directory."""
    from regulonado.scaling import infer_scale_factors, save_scale_factors

    bw_files = sorted(bigwig_dir.glob(glob))
    if not bw_files:
        typer.echo(f"No files matching '{glob}' found in {bigwig_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(bw_files)} bigwig files")
    df = infer_scale_factors(bw_files, max_workers=max_workers)
    save_scale_factors(df, output, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"Saved scale factors to {output}")


@app.command()
def calculate_original_scaling(
    metadata: Annotated[Path, typer.Argument(help="Path to regulonado_metadata.json")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path (default: <metadata_dir>/scale_factors.parquet)"),
    ] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="Output format: csv or parquet")] = "parquet",
    max_workers: Annotated[int, typer.Option("--workers", "-w", help="Thread pool size")] = 16,
) -> None:
    """Infer original scale factors for the final_bigwig_paths recorded in a dataset metadata file.

    Output rows are sorted by track_index so they can be applied directly by position.
    """
    import json

    import pandas as pd

    from regulonado.scaling import compute_clip_thresholds, infer_scale_factors, save_scale_factors

    if not metadata.exists():
        typer.echo(f"Metadata file not found: {metadata}", err=True)
        raise typer.Exit(1)

    with metadata.open() as fh:
        meta = json.load(fh)

    track_records = meta.get("final_track_records", [])
    if not track_records:
        typer.echo("No 'final_track_records' found in metadata.", err=True)
        raise typer.Exit(1)

    bin_size: int = int(meta.get("bin_size", 32))

    # Sort records by track_index to define the canonical order.
    track_records = sorted(track_records, key=lambda r: r["track_index"])
    bw_paths = [Path(r["resolved_path"]) for r in track_records]

    ext = "parquet" if fmt == "parquet" else "csv"
    out_path = output if output is not None else metadata.parent / f"scale_factors.{ext}"

    typer.echo(f"Metadata : {metadata}")
    typer.echo(f"Tracks   : {len(bw_paths)}")
    typer.echo(f"Bin size : {bin_size} bp")
    typer.echo(f"Output   : {out_path}")

    df = infer_scale_factors(bw_paths, max_workers=max_workers)

    # bamnado returns scale_factor = library_size / 1e9, which is the RPKM→raw-counts
    # factor without the bin_size term.  RPKM = reads / (lib/1e6) / (bin_size/1e3),
    # so raw_count = RPKM × (lib/1e6) × (bin_size/1e3) = RPKM × sf_bamnado × bin_size.
    df["scale_factor"] = df["scale_factor"] * bin_size

    # Join track_index and resolved_path from the records, then sort so row i
    # corresponds to track i — enabling direct positional application.
    records_df = pd.DataFrame(
        [{"track_index": r["track_index"], "resolved_path": r["resolved_path"]} for r in track_records]
    )
    df = df.merge(records_df, left_on="path", right_on="resolved_path", how="left")
    df = df.drop(columns=["resolved_path"]).sort_values("track_index").reset_index(drop=True)

    df = compute_clip_thresholds(df)

    # Put track_index first, then the fields consumed by train.py.
    priority = ["track_index", "scale_factor", "clip_soft", "clip_hard"]
    rest = [c for c in df.columns if c not in priority]
    df = df[priority + rest]

    save_scale_factors(df, out_path, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"Saved scale factors to {out_path}")
    typer.echo("Run 'regulonado enrich-metadata' to write these values into final_track_records.")


@app.command()
def calculate_tmm_scaling(
    metadata: Annotated[Path, typer.Argument(help="Path to regulonado_metadata.json")],
    scale_factors: Annotated[
        Optional[Path],
        typer.Option("--scale-factors", "-s", help="Scale-factors parquet from calculate-original-scaling (default: <metadata_dir>/scale_factors.parquet)"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path (default: overwrites --scale-factors input)"),
    ] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="Output format: csv or parquet")] = "parquet",
    split: Annotated[str, typer.Option("--split", help="Dataset split to use for TMM estimation")] = "train",
    trim_m: Annotated[float, typer.Option("--trim-m", help="Fraction to trim from each M-value tail (edgeR default 0.3)")] = 0.3,
    trim_a: Annotated[float, typer.Option("--trim-a", help="Fraction to trim from each A-value tail (edgeR default 0.05)")] = 0.05,
    min_count: Annotated[float, typer.Option("--min-count", help="Minimum pseudo-count for a region to be included")] = 1.0,
) -> None:
    """Compute edgeR-style TMM normalisation factors from the Arrow dataset.

    Reads per-sample mean RPKM from the Arrow shards under <metadata_dir>/<split>/,
    converts to pseudo-counts using library sizes from the scale-factors parquet,
    and runs TMM estimation over the full set of genomic regions.

    The output parquet gains a ``tmm_factor`` column and the ``scale_factor``
    column is updated to ``old_scale_factor / tmm_factor`` so that multiplying
    any raw RPKM BigWig value by the new scale_factor yields TMM-normalised
    approximate raw counts.

    Run ``regulonado enrich-metadata`` afterwards to write the updated values
    into ``final_track_records`` in the metadata JSON.

    \b
    Typical workflow
    ----------------
    # 1. Compute RPKM→raw-counts scale factors:
    regulonado calculate-original-scaling metadata.json

    # 2. Add TMM correction on top:
    regulonado calculate-tmm-scaling metadata.json

    # 3. Write back into final_track_records:
    regulonado enrich-metadata metadata.json scale_factors.parquet
    """
    import json

    import pandas as pd

    from regulonado.scaling import compute_tmm_factors, read_dataset_means, save_scale_factors

    if not metadata.exists():
        typer.echo(f"Metadata file not found: {metadata}", err=True)
        raise typer.Exit(1)

    with metadata.open() as fh:
        meta = json.load(fh)

    dataset_dir = metadata.parent
    bin_size: int = int(meta.get("bin_size", 32))

    ext = "parquet" if fmt == "parquet" else "csv"
    sf_path = scale_factors if scale_factors is not None else dataset_dir / f"scale_factors.{ext}"
    out_path = output if output is not None else sf_path

    if not sf_path.exists():
        typer.echo(
            f"Scale-factors file not found: {sf_path}\n"
            "Run 'regulonado calculate-original-scaling' first.",
            err=True,
        )
        raise typer.Exit(1)

    sf_df = pd.read_parquet(sf_path) if str(sf_path).endswith(".parquet") else pd.read_csv(sf_path)

    if "library_size" not in sf_df.columns:
        typer.echo("Column 'library_size' missing from scale-factors file.", err=True)
        raise typer.Exit(1)
    if "scale_factor" not in sf_df.columns:
        typer.echo("Column 'scale_factor' missing from scale-factors file.  Run calculate-original-scaling first.", err=True)
        raise typer.Exit(1)

    library_sizes = sf_df.sort_values("track_index")["library_size"].to_numpy(dtype=float)

    typer.echo(f"Dataset  : {dataset_dir}")
    typer.echo(f"Split    : {split}")
    typer.echo(f"Tracks   : {len(library_sizes)}")
    typer.echo(f"Bin size : {bin_size} bp")
    typer.echo("")

    means, n_tracks, n_bins = read_dataset_means(dataset_dir, split=split)

    if n_tracks != len(library_sizes):
        typer.echo(
            f"Track count mismatch: dataset has {n_tracks} tracks, "
            f"scale-factors file has {len(library_sizes)}.",
            err=True,
        )
        raise typer.Exit(1)

    region_length_kb = n_bins * bin_size / 1000.0
    typer.echo(f"Samples  : {means.shape[0]}")
    typer.echo(f"Bins/sample: {n_bins}  ({region_length_kb:.1f} kb)")
    typer.echo("")

    tmm = compute_tmm_factors(
        means,
        library_sizes,
        region_length_kb,
        trim_m=trim_m,
        trim_a=trim_a,
        min_count=min_count,
    )

    # Report
    sf_sorted = sf_df.sort_values("track_index").reset_index(drop=True)
    typer.echo(f"{'Track':>5}  {'samplename':<30}  {'tmm_factor':>12}  {'old_sf':>12}  {'new_sf':>12}")
    for i, (_, row) in enumerate(sf_sorted.iterrows()):
        old_sf = float(row["scale_factor"])
        new_sf = old_sf / tmm[i]
        name = str(row.get("samplename", i))[:30]
        typer.echo(f"{int(row['track_index']):>5}  {name:<30}  {tmm[i]:>12.6f}  {old_sf:>12.6f}  {new_sf:>12.6f}")

    # Write updated parquet: add tmm_factor, overwrite scale_factor
    sf_df = sf_df.sort_values("track_index").reset_index(drop=True)
    sf_df["tmm_factor"] = tmm
    sf_df["scale_factor"] = sf_df["scale_factor"] / sf_df["tmm_factor"]

    priority = ["track_index", "scale_factor", "tmm_factor", "clip_soft", "clip_hard"]
    rest = [c for c in sf_df.columns if c not in priority]
    sf_df = sf_df[priority + rest]

    save_scale_factors(sf_df, out_path, fmt=fmt)  # type: ignore[arg-type]
    typer.echo(f"\nSaved updated scale factors to {out_path}")
    typer.echo("Run 'regulonado enrich-metadata' to write these values into final_track_records.")


@app.command()
def enrich_metadata(
    metadata: Annotated[Path, typer.Argument(help="Path to regulonado_metadata.json to update in-place")],
    scale_factors: Annotated[Path, typer.Argument(help="Parquet (or CSV) produced by calculate-original-scaling")],
    fields: Annotated[
        Optional[list[str]],
        typer.Option("--field", "-f", help="Field to copy into final_track_records (repeat; default: all of scale_factor clip_soft clip_hard)"),
    ] = None,
) -> None:
    """Write scale_factor / clip_soft / clip_hard into final_track_records in a metadata JSON.

    Matches rows by track_index.  Writes the updated JSON back to the same file.
    train.py reads these fields from final_track_records, so running this command
    is the last step before training.
    """
    import json

    import pandas as pd

    fields_to_copy = list(fields) if fields else ["scale_factor", "clip_soft", "clip_hard"]

    if not metadata.exists():
        typer.echo(f"Metadata file not found: {metadata}", err=True)
        raise typer.Exit(1)
    if not scale_factors.exists():
        typer.echo(f"Scale-factors file not found: {scale_factors}", err=True)
        raise typer.Exit(1)

    sf_df = pd.read_parquet(scale_factors) if str(scale_factors).endswith(".parquet") else pd.read_csv(scale_factors)

    missing = [f for f in fields_to_copy if f not in sf_df.columns]
    if missing:
        typer.echo(f"Fields missing from scale-factors file: {missing}", err=True)
        raise typer.Exit(1)

    sf_by_idx: dict[int, dict] = {
        int(row["track_index"]): {f: row[f] for f in fields_to_copy}
        for _, row in sf_df.iterrows()
    }

    with metadata.open() as fh:
        meta = json.load(fh)

    records = meta.get("final_track_records", [])
    updated = 0
    for record in records:
        idx = int(record["track_index"])
        if idx in sf_by_idx:
            record.update({k: float(v) for k, v in sf_by_idx[idx].items()})
            updated += 1

    meta["final_track_records"] = records
    with metadata.open("w") as fh:
        json.dump(meta, fh, indent=2)

    typer.echo(f"Updated {updated}/{len(records)} track records in {metadata}")
    typer.echo(f"Fields written: {fields_to_copy}")


@app.command()
def build(
    bed_file: Annotated[
        Path, typer.Argument(help="BED file; column 4 used as fold label")
    ],
    fasta_file: Annotated[
        Path, typer.Argument(help="Reference genome FASTA (.fai index required)")
    ],
    output_dir: Annotated[
        Path, typer.Argument(help="Output directory for the Arrow DatasetDict")
    ],
    bigwig: Annotated[
        Optional[list[Path]],
        typer.Option(
            "--bigwig", "-b",
            help="BigWig file (repeat for each track, order is preserved)",
        ),
    ] = None,
    bigwig_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--bigwig-dir",
            help="Directory of BigWig files (sorted by name, alternative to --bigwig)",
        ),
    ] = None,
    bigwig_glob: Annotated[
        str, typer.Option("--bigwig-glob", help="Glob when using --bigwig-dir")
    ] = "*.bw",
    split: Annotated[
        Optional[list[str]],
        typer.Option(
            "--split", "-s",
            help=(
                "Split definition NAME:FOLD1,FOLD2 (repeat for each split). "
                "Default: train:fold0-2,fold5-7, validation:fold4, test:fold3."
            ),
        ),
    ] = None,
    context_length: Annotated[
        int, typer.Option("--context-length", help="Input sequence length in bp")
    ] = 524_288,
    bin_size: Annotated[
        int, typer.Option("--bin-size", help="Signal bin size in bp")
    ] = 32,
    n_pred_bins: Annotated[
        int, typer.Option("--n-pred-bins", help="Number of prediction bins")
    ] = 6_144,
    shift_max_bp: Annotated[
        int,
        typer.Option(
            "--shift-max-bp",
            help="Shift augmentation buffer on each side in bp (multiple of bin-size)",
        ),
    ] = 0,
    io_threads: Annotated[
        int,
        typer.Option("--io-threads", help="Threads for parallel BigWig reads per sample"),
    ] = 8,
    num_proc: Annotated[
        int, typer.Option("--num-proc", "-j", help="Parallel worker processes for generation")
    ] = 1,
    cache_dir: Annotated[
        Optional[str],
        typer.Option(
            "--cache-dir",
            help="Arrow cache dir (default: $SLURM_TMPDIR/hf_cache or /tmp/hf_cache)",
        ),
    ] = None,
    writer_batch_size: Annotated[
        int,
        typer.Option(
            "--writer-batch-size",
            help="Samples buffered before flushing an Arrow shard to scratch",
        ),
    ] = 500,
    stage: Annotated[
        bool,
        typer.Option(
            "--stage/--no-stage", help="Copy FASTA and BigWigs to scratch before building"
        ),
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Regenerate splits that already exist")
    ] = False,
    drop_missing: Annotated[
        bool, typer.Option("--drop-missing", help="Drop missing BigWig paths instead of raising an error")
    ] = False,
    dedupe_tracks: Annotated[
        str,
        typer.Option(
            "--dedupe-tracks",
            help="Track deduplication mode: none, identity, or content",
        ),
    ] = "none",
    profile: Annotated[
        bool,
        typer.Option(
            "--profile/--no-profile",
            help="Emit per-phase timing summary to stderr after each split (fast path only)",
        ),
    ] = False,
    no_fast_path: Annotated[
        bool,
        typer.Option(
            "--no-fast-path",
            help="Use the legacy pybigtools path instead of the two-phase Rust extraction",
        ),
    ] = False,
    n_extract_threads: Annotated[
        int,
        typer.Option(
            "--n-extract-threads",
            help="Rayon thread count for Phase 1 Rust BigWig extraction (fast path only)",
        ),
    ] = 32,
    signal_sample_chunk: Annotated[
        int,
        typer.Option(
            "--signal-sample-chunk",
            help="Samples per block when transposing extracted signals for Arrow writing",
        ),
    ] = 8,
    signal_track_chunk: Annotated[
        int,
        typer.Option(
            "--signal-track-chunk",
            help="Tracks per block when transposing extracted signals for Arrow writing",
        ),
    ] = 128,
    arrow_batch_size: Annotated[
        int,
        typer.Option(
            "--arrow-batch-size",
            help="Samples per Rust-written Arrow record batch",
        ),
    ] = 8,
    arrow_compression: Annotated[
        str,
        typer.Option(
            "--arrow-compression",
            help="Arrow IPC compression: zstd, lz4, or none",
        ),
    ] = "lz4",
    arrow_write_threads: Annotated[
        Optional[int],
        typer.Option(
            "--arrow-write-threads",
            help=(
                "Concurrent Arrow shard writers for chrom_pass. Defaults to "
                "min(8, --n-extract-threads); lower this if memory is tight."
            ),
        ),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            help=(
                "Build strategy: 'chrom_pass' (default, one shard per "
                "chromosome, ~10× fewer BigWig seeks) or 'fast' (sample-"
                "batched, single shard per split)."
            ),
        ),
    ] = "chrom_pass",
    chrom: Annotated[
        Optional[list[str]],
        typer.Option(
            "--chrom",
            help=(
                "Restrict each split to BED rows on this chromosome "
                "(repeat for several). The `index` column on every output "
                "row still refers to the absolute row in the full BED."
            ),
        ),
    ] = None,
) -> None:
    """Build an Arrow DatasetDict from BED / FASTA / BigWig sources.

    \b
    Examples
    --------
    # Two splits, 16 workers, shift aug, staged from Ceph scratch:
    regulonado build intervals.bed genome.fa out/ \\
        --bigwig-dir bw/ \\
        --split train:train --split validation:valid \\
        --shift-max-bp 128 --num-proc 16 --stage

    # Explicit ordered BigWig list, no fold filtering:
    regulonado build intervals.bed genome.fa out/ \\
        --bigwig plus.bw --bigwig minus.bw
    """
    from regulonado.dataset import DEFAULT_SPLITS, build_dataset, build_dataset_fast

    # --- resolve BigWig paths ------------------------------------------------
    if bigwig_dir is not None:
        bw_paths: list[str] = [str(p) for p in sorted(bigwig_dir.glob(bigwig_glob))]
        if not bw_paths:
            typer.echo(f"No files matching '{bigwig_glob}' in {bigwig_dir}", err=True)
            raise typer.Exit(1)
    elif bigwig:
        bw_paths = [str(p) for p in bigwig]
    else:
        typer.echo("Provide --bigwig files or --bigwig-dir.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Tracks : {len(bw_paths)}")

    # --- parse --split NAME:FOLD1,FOLD2 --------------------------------------
    splits: dict[str, list[str]] = {}
    for s in split or []:
        if ":" not in s:
            typer.echo(f"Invalid --split '{s}' — expected NAME:FOLD1,FOLD2", err=True)
            raise typer.Exit(1)
        name, _, folds_str = s.partition(":")
        splits[name] = [f.strip() for f in folds_str.split(",") if f.strip()]

    if not splits:
        splits = DEFAULT_SPLITS

    typer.echo(f"Splits : {list(splits)}")
    typer.echo(f"Output : {output_dir}")

    if no_fast_path:
        build_dataset(
            bed_file=bed_file,
            fasta_file=fasta_file,
            bigwig_paths=bw_paths,
            output_dir=output_dir,
            splits=splits,
            context_length=context_length,
            bin_size=bin_size,
            n_pred_bins=n_pred_bins,
            shift_max_bp=shift_max_bp,
            n_io_threads=io_threads,
            num_proc=num_proc,
            cache_dir=cache_dir,
            writer_batch_size=writer_batch_size,
            stage_to_scratch=stage,
            overwrite=overwrite,
            drop_missing=drop_missing,
            dedupe_tracks=dedupe_tracks,
            return_dataset=False,
        )
    else:
        build_dataset_fast(
            bed_file=bed_file,
            fasta_file=fasta_file,
            bigwig_paths=bw_paths,
            output_dir=output_dir,
            splits=splits,
            context_length=context_length,
            bin_size=bin_size,
            n_pred_bins=n_pred_bins,
            shift_max_bp=shift_max_bp,
            n_extract_threads=n_extract_threads,
            signal_sample_chunk=signal_sample_chunk,
            signal_track_chunk=signal_track_chunk,
            arrow_batch_size=arrow_batch_size,
            arrow_compression=arrow_compression,
            arrow_write_threads=arrow_write_threads,
            num_proc=num_proc,
            cache_dir=cache_dir,
            writer_batch_size=writer_batch_size,
            stage_to_scratch=stage,
            overwrite=overwrite,
            drop_missing=drop_missing,
            dedupe_tracks=dedupe_tracks,
            profile=profile,
            strategy=strategy,
            chrom_filter=list(chrom) if chrom else None,
            return_dataset=False,
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
