//! Chromosome-pass Parquet writer.
//!
//! For each chromosome, decode the binned signal of all tracks once into an
//! in-RAM `(n_tracks, n_chrom_bins)` matrix, then slice per-sample rows out of
//! that matrix. This collapses ~N_samples random BigWig seeks per chromosome
//! into one sequential pass per (chrom, track) pair.
//!
//! Output: Parquet shards `{split}-NNNNN-of-MMMMM.parquet`, all splits sharing
//! one output directory, ordered chrom-major with shard `00000` on the longest
//! chromosome that has at least one sample in this split (descending by
//! length). Each chromosome is split into `ceil(samples / shard_size)` shard
//! files. Rows within a shard are written one at a time (bounding writer RAM
//! to one example), grouped into Parquet row groups of `rows_per_row_group`
//! rows. Rows within a shard are in original BED order. The schema includes a
//! `local_index` column so downstream code can recover original BED order via
//! `dataset.sort("index")` if needed.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{
    builder::{Int64Builder, StringBuilder},
    ArrayRef, RecordBatch,
};
use parquet::arrow::ArrowWriter;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::arrow_schema::{sequence_tokens_array, signal_array, window_arrow_schema};
use crate::bigwig_io::{open_bigwig_handles, BwHandle};
use crate::binning::{bin_region_into, BinningOptions, BinningScratch, BinningUsage};
use crate::fasta::{load_fasta_index, read_sequence_tokens};
use crate::io_utils::{maybe_log_progress, parquet_writer_properties};

/// Timing profile for a single Parquet shard write.
///
/// Tracks raw and wall-clock time spent in each stage of writing one shard
/// file, along with row and byte counts. Aggregated across shards for
/// per-stage analysis when profiling is enabled.
#[derive(Clone, Copy, Debug, Default)]
struct WriteShardProfile {
    rows: usize,
    bytes: u64,
    slice_ns: u128,
    fasta_ns: u128,
    row_build_ns: u128,
    parquet_write_ns: u128,
    wall_ns: u128,
}

impl WriteShardProfile {
    fn add(&mut self, other: WriteShardProfile) {
        self.rows += other.rows;
        self.bytes += other.bytes;
        self.slice_ns += other.slice_ns;
        self.fasta_ns += other.fasta_ns;
        self.row_build_ns += other.row_build_ns;
        self.parquet_write_ns += other.parquet_write_ns;
        self.wall_ns += other.wall_ns;
    }
}

/// Specification for writing a single Parquet shard file.
///
/// Describes the split name (for the `{split}-NNNNN-of-MMMMM.parquet` filename),
/// shard numbering, and the range of samples within the split that this shard
/// covers.
struct WriteShardSpec<'a> {
    split_name: &'a str,
    out_dir: &'a str,
    shard_idx: usize,
    shard_total: usize,
    /// Sample range [shard_start, shard_end) covered by this shard file.
    shard_start: usize,
    shard_end: usize,
    samples: &'a [(usize, usize)],
}

/// Context passed to each shard-write task.
///
/// Contains references to all shared data (schema, decoded chromosome signals,
/// FASTA index, BED rows) and settings (bin/context dimensions, writer
/// properties). Each worker thread (via `write_chrom_shard`) reads from this
/// context to construct its shard file without copying the large signal matrix.
struct WriteShardCtx<'a> {
    schema: Arc<arrow_schema::Schema>,
    rows_per_row_group: usize,
    zstd_level: i32,
    chrom_signals: &'a [f32],
    n_chrom_bins: usize,
    n_tracks: usize,
    n_bins: usize,
    context_len: usize,
    bin_size: u32,
    signal_intervals: &'a [(String, u32, u32)],
    bed_rows: &'a [(String, u32, u32, String)],
    fasta_path: &'a str,
    fai: &'a HashMap<String, crate::fasta::FastaIndexRecord>,
    missing_value: f32,
}

/// Write a single Parquet shard file for one chromosome.
///
/// For each sample in the shard, slices the decoded chromosome signal matrix to extract
/// the binned labels for that sample's interval, reads its FASTA sequence tokens, and
/// appends one row to the Parquet writer. The decoded signal was already read once during
/// the per-chromosome scan phase; this function only slices per-sample rows from it.
///
/// Returns a profile with timing breakdowns and byte count for later aggregation,
/// or a String error if file I/O, Arrow construction, or FASTA read fails.
fn write_chrom_shard(
    spec: &WriteShardSpec<'_>,
    ctx: &WriteShardCtx<'_>,
) -> Result<WriteShardProfile, String> {
    let started = Instant::now();
    let rows_in_shard = spec.shard_end - spec.shard_start;
    let shard_path = format!(
        "{}/{}-{:05}-of-{:05}.parquet",
        spec.out_dir, spec.split_name, spec.shard_idx, spec.shard_total
    );

    let fasta = std::fs::File::open(ctx.fasta_path)
        .map_err(|e| format!("Cannot open FASTA {}: {e}", ctx.fasta_path))?;
    let out_file = std::fs::File::create(&shard_path)
        .map_err(|e| format!("Cannot create Parquet file {shard_path}: {e}"))?;
    let props = parquet_writer_properties(ctx.rows_per_row_group, ctx.zstd_level)?;
    let mut writer = ArrowWriter::try_new(out_file, Arc::clone(&ctx.schema), Some(props))
        .map_err(|e| e.to_string())?;

    let mut slice_ns: u128 = 0;
    let mut fasta_ns: u128 = 0;
    let mut row_build_ns: u128 = 0;
    let mut parquet_write_ns: u128 = 0;

    // Write one sample per row, so writer RAM is bounded by a single example
    // regardless of shard size. Parquet groups rows into row groups of
    // `rows_per_row_group` internally via the writer properties.
    for (local_idx, global_idx) in spec.samples[spec.shard_start..spec.shard_end]
        .iter()
        .copied()
    {
        let t_slice = Instant::now();
        // Bins past the chromosome's last whole bin are padding: missing, not zero signal.
        let mut label = vec![ctx.missing_value; ctx.n_tracks * ctx.n_bins];
        let (_chrom, sig_start, sig_end) = &ctx.signal_intervals[global_idx];
        let bin_start = (*sig_start / ctx.bin_size) as usize;
        let bin_end_raw = (*sig_end / ctx.bin_size) as usize;
        let bin_end = bin_end_raw.min(ctx.n_chrom_bins);
        let copy_n = bin_end.saturating_sub(bin_start).min(ctx.n_bins);

        if copy_n > 0 && bin_start < ctx.n_chrom_bins {
            for track_idx in 0..ctx.n_tracks {
                let src_start = track_idx * ctx.n_chrom_bins + bin_start;
                let dst_start = track_idx * ctx.n_bins;
                label[dst_start..dst_start + copy_n]
                    .copy_from_slice(&ctx.chrom_signals[src_start..src_start + copy_n]);
            }
        }
        slice_ns += t_slice.elapsed().as_nanos();

        let t_fasta = Instant::now();
        let (bed_chrom, bed_start, bed_end, _) = &ctx.bed_rows[global_idx];
        let tokens = read_sequence_tokens(
            &fasta,
            ctx.fai,
            bed_chrom,
            *bed_start,
            *bed_end,
            ctx.context_len,
        )?;
        fasta_ns += t_fasta.elapsed().as_nanos();

        let t_row = Instant::now();
        let mut interval_builder = StringBuilder::with_capacity(1, 32);
        let mut index_builder = Int64Builder::with_capacity(1);
        let mut local_index_builder = Int64Builder::with_capacity(1);
        interval_builder.append_value(format!("{bed_chrom}:{bed_start}-{bed_end}"));
        index_builder.append_value(global_idx as i64);
        local_index_builder.append_value(local_idx as i64);

        let batch = RecordBatch::try_new(
            Arc::clone(&ctx.schema),
            vec![
                sequence_tokens_array(tokens, ctx.context_len),
                signal_array(label, ctx.n_tracks, ctx.n_bins),
                Arc::new(interval_builder.finish()) as ArrayRef,
                Arc::new(index_builder.finish()) as ArrayRef,
                Arc::new(local_index_builder.finish()) as ArrayRef,
            ],
        )
        .map_err(|e| e.to_string())?;
        row_build_ns += t_row.elapsed().as_nanos();

        let t_write = Instant::now();
        writer.write(&batch).map_err(|e| e.to_string())?;
        parquet_write_ns += t_write.elapsed().as_nanos();
    }

    let t_finish = Instant::now();
    writer.close().map_err(|e| e.to_string())?;
    parquet_write_ns += t_finish.elapsed().as_nanos();

    let bytes = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
    Ok(WriteShardProfile {
        rows: rows_in_shard,
        bytes,
        slice_ns,
        fasta_ns,
        row_build_ns,
        parquet_write_ns,
        wall_ns: started.elapsed().as_nanos(),
    })
}

/// State for writing one split's shards across all chromosomes.
///
/// Tracks which samples belong to this split (by chromosome), how many shards total
/// will be written, and the next shard index to assign. Updated as each chromosome's
/// shards are written to produce monotonically increasing `{split}-NNNNN-of-MMMMM.parquet`
/// filenames.
struct SplitChromSamples {
    name: String,
    total_samples: usize,
    samples_by_chrom: HashMap<String, Vec<(usize, usize)>>,
    total_shards: usize,
    next_shard: usize,
}

/// Per-split sample grouping paired with the chromosome scan order.
///
/// The second element is `(chromosome_name, length_bp)` sorted by descending
/// length, which fixes the order chromosomes are scanned in.
type SplitPlan = (Vec<SplitChromSamples>, Vec<(String, u64)>);

/// Build split-wide state and determine chromosome scan order.
///
/// Validates that split_names and split_sample_indices have the same length, then
/// groups each split's samples by chromosome. Returns both a per-split state struct
/// (holding samples_by_chrom and total_shards for progress tracking) and a sorted
/// list of chromosomes ordered descending by length (so the longest chromosome
/// lands in shard `00000`). Chromosomes with zero bins after division by bin_size
/// are excluded from the scan.
fn build_split_chrom_samples(
    split_names: Vec<String>,
    split_sample_indices: Vec<Vec<usize>>,
    bed_rows: &[(String, u32, u32, String)],
    chrom_lengths: &HashMap<String, crate::fasta::FastaIndexRecord>,
    bin_size: u32,
    shard_size: usize,
) -> Result<SplitPlan, String> {
    if split_names.len() != split_sample_indices.len() {
        return Err(format!(
            "split_names and split_sample_indices must have the same length (got {}, {})",
            split_names.len(),
            split_sample_indices.len(),
        ));
    }

    let mut chrom_seen: HashMap<String, u64> = HashMap::new();
    let mut splits = Vec::with_capacity(split_names.len());
    for (name, sample_indices) in split_names.into_iter().zip(split_sample_indices) {
        let mut samples_by_chrom: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        for (local_idx, global_idx) in sample_indices.iter().copied().enumerate() {
            let chrom = bed_rows
                .get(global_idx)
                .map(|r| r.0.clone())
                .ok_or_else(|| format!("BED index {global_idx} is out of range"))?;
            samples_by_chrom
                .entry(chrom.clone())
                .or_default()
                .push((local_idx, global_idx));
            let len = chrom_lengths.get(&chrom).map(|r| r.len).unwrap_or(0);
            if ((len as usize) / (bin_size as usize)) > 0 {
                chrom_seen.entry(chrom).or_insert(len);
            }
        }

        let total_shards = samples_by_chrom
            .iter()
            .filter(|(chrom, _)| {
                chrom_lengths
                    .get(*chrom)
                    .map(|r| ((r.len as usize) / (bin_size as usize)) > 0)
                    .unwrap_or(false)
            })
            .map(|(_, samples)| samples.len().div_ceil(shard_size))
            .sum();

        splits.push(SplitChromSamples {
            name,
            total_samples: sample_indices.len(),
            samples_by_chrom,
            total_shards,
            next_shard: 0,
        });
    }

    let mut chrom_order: Vec<(String, u64)> = chrom_seen.into_iter().collect();
    chrom_order.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok((splits, chrom_order))
}

/// Write Parquet datasets using the chromosome-pass strategy.
///
/// Scans each chromosome once, decoding the binned signal of all tracks into a
/// single (n_tracks, n_chrom_bins) matrix in RAM, then slices per-sample rows out of
/// that matrix. This collapses ~N_samples random BigWig seeks per chromosome into one
/// sequential scan per (chrom, track) pair, achieving much better BigWig I/O locality.
///
/// All splits share a single chromosome scan and a single output directory. For each
/// split, writes one Parquet shard file per chromosome per shard (ordered chrom-major,
/// with shard `00000` on the longest chromosome), named `{split}-NNNNN-of-MMMMM.parquet`.
/// Within a shard, rows are written one at a time (bounding writer RAM to a single
/// example) and grouped into row groups of `rows_per_row_group` rows by the Parquet
/// writer. Rows within a shard are in original BED order; a `local_index` column
/// preserves the original position for sorting.
///
/// Releases the GIL during the per-chromosome scan and during shard file writing, and
/// fans out over tracks within each chromosome using Rayon parallelism.
///
/// # Parameters
///
/// - `bw_paths`: paths to BigWig files, one per track.
/// - `minus_flags`: boolean flags indicating whether each track's values should be
///   negated if the majority (≥80%) of non-zero values are already negative.
/// - `signal_intervals`: list of (chrom, region_start, region_end) tuples defining
///   the binned signal region for each sample, aligned to bin boundaries.
/// - `split_names`: names of the splits being written (train, validation, test, etc.).
/// - `out_dir`: shared output directory where all splits' Parquet shards are written.
/// - `split_sample_indices`: for each split, a list of indices into `bed_rows` and
///   `signal_intervals` specifying which samples belong to that split.
/// - `bed_rows`: list of (chrom, start, end, name) tuples from the BED file; defines
///   sample intervals and metadata.
/// - `fasta_path`: path to a .fasta file with a corresponding .fasta.fai index.
/// - `n_bins`: number of bins in the signal output.
/// - `context_len`: length of the DNA context window (`stored_context`).
/// - `bin_size`: size of each bin in basepairs; used to convert between BED coordinates
///   and bin indices.
/// - `hf_features_json`: `datasets.Features.to_dict()` JSON, stored verbatim as the
///   `huggingface` schema metadata.
/// - `shard_size`: rows per Parquet shard file (default 0, meaning one shard per
///   chromosome per split).
/// - `rows_per_row_group`: rows per Parquet row group (default 1).
/// - `zstd_level`: explicit ZSTD compression level (default 3).
/// - `n_threads`: Rayon thread pool size for BigWig scanning and track parallelism.
///   If unset, uses available cores.
/// - `write_threads`: thread pool size for parallel shard writes (default 8 or
///   n_threads, whichever is smaller).
/// - `profile`: if true, collect and log timing breakdowns per stage.
/// - `mean_over_covered_bases`: divide each bin by its recorded bases rather than its
///   in-contig width (default false; see `BinningOptions`).
/// - `missing_as_nan`: write NaN for bins past the contig end or wholly NaN-valued
///   (default true); 0.0 otherwise.
///
/// # Returns
///
/// A mapping from split name to the number of rows written, for the dataset card.
///
/// # Errors
///
/// Returns a PyRuntimeError if:
/// - The FASTA file cannot be opened or the .fai index is missing.
/// - A BED index falls outside the bed_rows array.
/// - `zstd_level` is not a valid ZSTD compression level.
#[pyfunction]
#[pyo3(signature = (
    bw_paths,
    minus_flags,
    signal_intervals,
    split_names,
    out_dir,
    split_sample_indices,
    bed_rows,
    fasta_path,
    n_bins,
    context_len,
    bin_size,
    hf_features_json,
    shard_size=0,
    rows_per_row_group=1,
    zstd_level=3,
    n_threads=None,
    write_threads=None,
    profile=false,
    mean_over_covered_bases=false,
    missing_as_nan=true
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_parquet_splits_chrom_pass(
    py: Python<'_>,
    bw_paths: Vec<String>,
    minus_flags: Vec<bool>,
    signal_intervals: Vec<(String, u32, u32)>,
    split_names: Vec<String>,
    out_dir: String,
    split_sample_indices: Vec<Vec<usize>>,
    bed_rows: Vec<(String, u32, u32, String)>,
    fasta_path: String,
    n_bins: usize,
    context_len: usize,
    bin_size: u32,
    hf_features_json: String,
    shard_size: usize,
    rows_per_row_group: usize,
    zstd_level: i32,
    n_threads: Option<usize>,
    write_threads: Option<usize>,
    profile: bool,
    mean_over_covered_bases: bool,
    missing_as_nan: bool,
) -> PyResult<HashMap<String, usize>> {
    let binning_options = BinningOptions {
        mean_over_covered_bases,
        missing_as_nan,
    };
    crate::io_utils::configure_global_rayon(n_threads);

    let n_tracks = bw_paths.len();
    let rows_per_row_group = rows_per_row_group.max(1);
    let shard_size = if shard_size == 0 {
        usize::MAX
    } else {
        shard_size
    };

    let schema = window_arrow_schema(context_len, n_tracks, n_bins, hf_features_json);
    let fai = load_fasta_index(&fasta_path).map_err(PyRuntimeError::new_err)?;
    let (mut splits, chrom_order) = build_split_chrom_samples(
        split_names,
        split_sample_indices,
        &bed_rows,
        &fai,
        bin_size,
        shard_size,
    )
    .map_err(PyRuntimeError::new_err)?;

    let write_threads = write_threads
        .unwrap_or_else(|| n_threads.unwrap_or(8).min(8))
        .max(1);
    let write_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(write_threads)
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("Cannot build Parquet write pool: {e}")))?;

    let total_samples: usize = splits.iter().map(|s| s.total_samples).sum();
    let total_shards: usize = splits.iter().map(|s| s.total_shards).sum();
    let split_summary = splits
        .iter()
        .map(|s| {
            format!(
                "{}={} samples/{} shards",
                s.name, s.total_samples, s.total_shards
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!(
        "[regulonado_rs] chrom_pass(all_splits): {} tracks × {} chromosomes ({} samples, {} shards) → [{}] (zstd_level={}, rows_per_row_group={}, write_threads={})",
        n_tracks,
        chrom_order.len(),
        total_samples,
        total_shards,
        split_summary,
        zstd_level,
        rows_per_row_group,
        write_threads,
    );

    let mut handles: Vec<BwHandle> = py
        .allow_threads(|| open_bigwig_handles(&bw_paths))
        .map_err(PyRuntimeError::new_err)?;

    let mut prof_chrom_scan_s: Vec<f64> = Vec::new();
    let mut prof_binning = BinningUsage::default();
    let mut prof_writer_wall_s: Vec<f64> = Vec::new();
    let mut prof_write = WriteShardProfile::default();

    let started = Instant::now();
    let mut last_log = started;
    let mut done_samples = 0usize;

    for (chrom_idx, (chrom, chrom_len_bp)) in chrom_order.iter().enumerate() {
        py.check_signals()?;
        let n_chrom_bins = (*chrom_len_bp as usize) / (bin_size as usize);
        let region_end = (n_chrom_bins as u32).saturating_mul(bin_size);
        if n_chrom_bins == 0 {
            eprintln!(
                "[regulonado_rs] chrom_pass(all_splits)[{chrom_idx}/{}] skipping chrom={chrom} (len {chrom_len_bp} < bin_size)",
                chrom_order.len(),
            );
            continue;
        }

        let chrom_shards: usize = splits
            .iter()
            .filter_map(|split| split.samples_by_chrom.get(chrom))
            .map(|samples| samples.len().div_ceil(shard_size))
            .sum();
        let chrom_samples: usize = splits
            .iter()
            .filter_map(|split| split.samples_by_chrom.get(chrom))
            .map(|samples| samples.len())
            .sum();
        eprintln!(
            "[regulonado_rs] chrom_pass(all_splits)[{chrom_idx}/{}] chrom={chrom} samples={} chrom_bins={} region_end={}bp output_shards={}",
            chrom_order.len(),
            chrom_samples,
            n_chrom_bins,
            region_end,
            chrom_shards,
        );

        let t_scan = Instant::now();
        let mut chrom_signals: Vec<f32> = vec![0.0; n_tracks * n_chrom_bins];
        let chrom_str: &str = chrom.as_str();

        let scan_usage = py.allow_threads(|| {
            chrom_signals
                .par_chunks_mut(n_chrom_bins)
                .zip(handles.par_iter_mut())
                .enumerate()
                .map_init(
                    BinningScratch::default,
                    |scratch, (track_idx, (out_row, reader))| {
                        let usage =
                            bin_region_into(
                                reader,
                                chrom_str,
                                0,
                                region_end,
                                out_row,
                                scratch,
                                binning_options,
                            );

                        let is_minus = minus_flags.get(track_idx).copied().unwrap_or(false);
                        if is_minus {
                            let mut nz = 0usize;
                            let mut neg = 0usize;
                            for &v in out_row.iter() {
                                if v != 0.0 && !v.is_nan() {
                                    nz += 1;
                                    if v < 0.0 {
                                        neg += 1;
                                    }
                                }
                            }
                            if nz > 0 && (neg as f32) / (nz as f32) >= 0.8 {
                                for v in out_row.iter_mut() {
                                    *v = -*v;
                                }
                            }
                        }
                        usage
                    },
                )
                .collect::<Result<Vec<_>, _>>()
        });
        let scan_usage = scan_usage.map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        if profile {
            prof_chrom_scan_s.push(t_scan.elapsed().as_secs_f64());
            for usage in scan_usage {
                prof_binning.add(usage);
            }
        }

        let t_writer = Instant::now();
        let mut write_specs: Vec<WriteShardSpec<'_>> = Vec::new();
        for split in splits.iter_mut() {
            let Some(samples) = split.samples_by_chrom.get(chrom) else {
                continue;
            };
            for shard_start in (0..samples.len()).step_by(shard_size) {
                write_specs.push(WriteShardSpec {
                    split_name: &split.name,
                    out_dir: &out_dir,
                    shard_idx: split.next_shard,
                    shard_total: split.total_shards,
                    shard_start,
                    shard_end: (shard_start + shard_size).min(samples.len()),
                    samples,
                });
                split.next_shard += 1;
            }
        }

        let write_ctx = WriteShardCtx {
            schema: Arc::clone(&schema),
            rows_per_row_group,
            zstd_level,
            chrom_signals: &chrom_signals,
            n_chrom_bins,
            n_tracks,
            n_bins,
            context_len,
            bin_size,
            signal_intervals: &signal_intervals,
            bed_rows: &bed_rows,
            fasta_path: &fasta_path,
            fai: &fai,
            missing_value: binning_options.missing_value(),
        };
        let shard_profiles: Vec<WriteShardProfile> = py
            .allow_threads(|| {
                write_pool.install(|| {
                    write_specs
                        .par_iter()
                        .map(|spec| write_chrom_shard(spec, &write_ctx))
                        .collect::<Result<Vec<_>, _>>()
                })
            })
            .map_err(PyRuntimeError::new_err)?;
        let writer_wall_s = t_writer.elapsed().as_secs_f64();

        for shard_profile in shard_profiles {
            done_samples += shard_profile.rows;
            if profile {
                prof_write.add(shard_profile);
            }
            maybe_log_progress(
                &mut last_log,
                started,
                "chrom_pass(all_splits) samples",
                done_samples,
                total_samples,
            );
        }
        if profile {
            prof_writer_wall_s.push(writer_wall_s);
        }
    }

    if profile {
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        eprintln!(
            "[regulonado_rs] chrom_pass(all_splits) profile chroms={}\n  \
             chrom_scan: total_wall_s={:.1}\n  \
             binner:     direct_calls={} direct_intervals={}\n  \
             writer:     total_wall_s={:.1} summed_worker_s={:.1} threads={}\n  \
             slice:      total_worker_s={:.1}\n  \
             fasta:      total_worker_s={:.1}\n  \
             row_build:  total_worker_s={:.1}\n  \
             parquet_write: total_worker_s={:.1}\n  \
             bytes:      {:.1} GiB\n  \
             throughput: {:.2} samples/s\n  \
             total_wall_s={:.1}",
            chrom_order.len(),
            sum(&prof_chrom_scan_s),
            prof_binning.direct_calls,
            prof_binning.direct_intervals,
            sum(&prof_writer_wall_s),
            prof_write.wall_ns as f64 / 1e9,
            write_threads,
            prof_write.slice_ns as f64 / 1e9,
            prof_write.fasta_ns as f64 / 1e9,
            prof_write.row_build_ns as f64 / 1e9,
            prof_write.parquet_write_ns as f64 / 1e9,
            prof_write.bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            prof_write.rows as f64 / started.elapsed().as_secs_f64().max(1e-9),
            started.elapsed().as_secs_f64(),
        );
    }

    let mut row_counts = HashMap::with_capacity(splits.len());
    for split in &splits {
        if split.next_shard != split.total_shards {
            return Err(PyRuntimeError::new_err(format!(
                "Split '{}' wrote {} shard(s), expected {}",
                split.name, split.next_shard, split.total_shards
            )));
        }
        row_counts.insert(split.name.clone(), split.total_samples);
    }

    Ok(row_counts)
}
