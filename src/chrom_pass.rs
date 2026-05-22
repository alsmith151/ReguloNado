//! Chromosome-pass Arrow writer.
//!
//! For each chromosome, decode the binned signal of all tracks once into an
//! in-RAM `(n_tracks, n_chrom_bins)` matrix, then slice per-sample rows out of
//! that matrix. This collapses ~N_samples random BigWig seeks per chromosome
//! into one sequential pass per (chrom, track) pair.
//!
//! Output: one Arrow IPC shard per chromosome, `data-NNNNN-of-MMMMM.arrow`,
//! where shard `00000` is the longest chromosome with at least one sample in
//! this split (descending by length). Rows within a shard are in original
//! BED order. The schema includes a `local_index` column so downstream code can
//! recover original BED order via `dataset.sort("index")` if needed.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{
    builder::{Int64Builder, StringBuilder},
    ArrayRef, RecordBatch,
};
use arrow_ipc::writer::StreamWriter;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::bigwig_io::{open_bigwig_handles, BwHandle};
use crate::binning::bin_region_into;
use crate::fasta::{load_fasta_index, read_one_hot_sequence};
use crate::io_utils::{ipc_write_options, maybe_log_progress};
use crate::schema::{hf_arrow_schema, make_2d_f32_array, make_2d_i8_array};

#[derive(Clone, Copy, Debug, Default)]
struct WriteShardProfile {
    rows: usize,
    bytes: u64,
    slice_ns: u128,
    fasta_ns: u128,
    batch_ns: u128,
    arrow_write_ns: u128,
    wall_ns: u128,
}

impl WriteShardProfile {
    fn add(&mut self, other: WriteShardProfile) {
        self.rows += other.rows;
        self.bytes += other.bytes;
        self.slice_ns += other.slice_ns;
        self.fasta_ns += other.fasta_ns;
        self.batch_ns += other.batch_ns;
        self.arrow_write_ns += other.arrow_write_ns;
        self.wall_ns += other.wall_ns;
    }
}

struct WriteShardSpec<'a> {
    out_dir: &'a str,
    shard_idx: usize,
    shard_total: usize,
    batch_start: usize,
    batch_end: usize,
    samples: &'a [(usize, usize)],
}

struct WriteShardCtx<'a> {
    schema: Arc<arrow_schema::Schema>,
    compression: &'a str,
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
}

fn write_chrom_shard(
    spec: &WriteShardSpec<'_>,
    ctx: &WriteShardCtx<'_>,
) -> Result<WriteShardProfile, String> {
    let started = Instant::now();
    let rows_in_batch = spec.batch_end - spec.batch_start;
    let shard_path = format!(
        "{}/data-{:05}-of-{:05}.arrow",
        spec.out_dir, spec.shard_idx, spec.shard_total
    );

    let fasta = std::fs::File::open(ctx.fasta_path)
        .map_err(|e| format!("Cannot open FASTA {}: {e}", ctx.fasta_path))?;
    let out_file = std::fs::File::create(&shard_path)
        .map_err(|e| format!("Cannot create Arrow file {shard_path}: {e}"))?;
    let write_options = ipc_write_options(ctx.compression)?;
    let mut writer = StreamWriter::try_new_with_options(out_file, &ctx.schema, write_options)
        .map_err(|e| e.to_string())?;

    let t_slice = Instant::now();
    let mut labels: Vec<f32> = vec![0.0; rows_in_batch * ctx.n_tracks * ctx.n_bins];
    for (row_idx, (_, global_idx)) in spec.samples[spec.batch_start..spec.batch_end]
        .iter()
        .copied()
        .enumerate()
    {
        let (_chrom, sig_start, sig_end) = &ctx.signal_intervals[global_idx];
        let bin_start = (*sig_start / ctx.bin_size) as usize;
        let bin_end_raw = (*sig_end / ctx.bin_size) as usize;
        let bin_end = bin_end_raw.min(ctx.n_chrom_bins);
        let copy_n = bin_end.saturating_sub(bin_start).min(ctx.n_bins);

        if copy_n > 0 && bin_start < ctx.n_chrom_bins {
            let row_base = row_idx * ctx.n_tracks * ctx.n_bins;
            for track_idx in 0..ctx.n_tracks {
                let src_start = track_idx * ctx.n_chrom_bins + bin_start;
                let dst_start = row_base + track_idx * ctx.n_bins;
                labels[dst_start..dst_start + copy_n]
                    .copy_from_slice(&ctx.chrom_signals[src_start..src_start + copy_n]);
            }
        }
    }
    let slice_ns = t_slice.elapsed().as_nanos();

    let t_fasta = Instant::now();
    let mut input_values = Vec::with_capacity(rows_in_batch * 4 * ctx.context_len);
    let mut interval_builder = StringBuilder::with_capacity(rows_in_batch, rows_in_batch * 32);
    let mut index_builder = Int64Builder::with_capacity(rows_in_batch);
    let mut local_index_builder = Int64Builder::with_capacity(rows_in_batch);

    for (local_idx, global_idx) in spec.samples[spec.batch_start..spec.batch_end]
        .iter()
        .copied()
    {
        let (bed_chrom, bed_start, bed_end, _) = &ctx.bed_rows[global_idx];
        let seq = read_one_hot_sequence(
            &fasta,
            ctx.fai,
            bed_chrom,
            *bed_start,
            *bed_end,
            ctx.context_len,
        )?;
        input_values.extend_from_slice(&seq);
        interval_builder.append_value(format!("{bed_chrom}:{bed_start}-{bed_end}"));
        index_builder.append_value(global_idx as i64);
        local_index_builder.append_value(local_idx as i64);
    }
    let fasta_ns = t_fasta.elapsed().as_nanos();

    let t_batch = Instant::now();
    let batch = RecordBatch::try_new(
        Arc::clone(&ctx.schema),
        vec![
            make_2d_i8_array(input_values, rows_in_batch, 4, ctx.context_len),
            make_2d_f32_array(labels, rows_in_batch, ctx.n_tracks, ctx.n_bins),
            Arc::new(interval_builder.finish()) as ArrayRef,
            Arc::new(index_builder.finish()) as ArrayRef,
            Arc::new(local_index_builder.finish()) as ArrayRef,
        ],
    )
    .map_err(|e| e.to_string())?;
    let batch_ns = t_batch.elapsed().as_nanos();

    let t_arrow = Instant::now();
    writer.write(&batch).map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())?;
    let arrow_write_ns = t_arrow.elapsed().as_nanos();

    let bytes = std::fs::metadata(&shard_path).map(|m| m.len()).unwrap_or(0);
    Ok(WriteShardProfile {
        rows: rows_in_batch,
        bytes,
        slice_ns,
        fasta_ns,
        batch_ns,
        arrow_write_ns,
        wall_ns: started.elapsed().as_nanos(),
    })
}

#[pyfunction]
#[pyo3(signature = (
    bw_paths,
    minus_flags,
    signal_intervals,
    out_dir,
    sample_indices,
    bed_rows,
    fasta_path,
    n_bins,
    context_len,
    bin_size,
    batch_size=4,
    n_threads=None,
    arrow_write_threads=None,
    compression=None,
    profile=false
))]
pub(crate) fn write_arrow_split_chrom_pass(
    py: Python<'_>,
    bw_paths: Vec<String>,
    minus_flags: Vec<bool>,
    signal_intervals: Vec<(String, u32, u32)>,
    out_dir: String,
    sample_indices: Vec<usize>,
    bed_rows: Vec<(String, u32, u32, String)>,
    fasta_path: String,
    n_bins: usize,
    context_len: usize,
    bin_size: u32,
    batch_size: usize,
    n_threads: Option<usize>,
    arrow_write_threads: Option<usize>,
    compression: Option<String>,
    profile: bool,
) -> PyResult<()> {
    if let Some(nt) = n_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build_global()
            .ok();
    }

    let n_tracks = bw_paths.len();
    let schema = hf_arrow_schema(context_len, n_tracks, n_bins);

    let fai = load_fasta_index(&fasta_path).map_err(PyRuntimeError::new_err)?;
    // Group samples by chromosome, preserving original BED order within each chrom.
    let mut samples_by_chrom: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for (local_idx, global_idx) in sample_indices.iter().copied().enumerate() {
        let chrom = bed_rows
            .get(global_idx)
            .map(|r| r.0.clone())
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("BED index {global_idx} is out of range"))
            })?;
        samples_by_chrom
            .entry(chrom)
            .or_default()
            .push((local_idx, global_idx));
    }

    // Process chromosomes in descending order of length (fail fast on the largest).
    let mut chrom_order: Vec<(String, u64)> = samples_by_chrom
        .keys()
        .map(|c| {
            let len = fai.get(c).map(|r| r.len).unwrap_or(0);
            (c.clone(), len)
        })
        .collect();
    chrom_order.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let n_chroms_used = chrom_order.len();
    let compression = compression.unwrap_or_else(|| "zstd".to_string());
    let write_threads = arrow_write_threads
        .unwrap_or_else(|| n_threads.unwrap_or(8).min(8))
        .max(1);
    let write_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(write_threads)
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("Cannot build Arrow write pool: {e}")))?;

    eprintln!(
        "[regulonado_rs] chrom_pass: {} tracks × {} chromosomes ({} samples) → {} (compression={}, arrow_write_threads={})",
        n_tracks,
        n_chroms_used,
        sample_indices.len(),
        out_dir,
        compression,
        write_threads,
    );

    // Validate Arrow i32 offset constraints once.
    let batch_size = batch_size.max(1);
    let max_label_offset = batch_size.saturating_mul(n_tracks).saturating_mul(n_bins);
    let max_input_offset = batch_size.saturating_mul(4).saturating_mul(context_len);
    if max_label_offset > i32::MAX as usize || max_input_offset > i32::MAX as usize {
        let safe_batch = (i32::MAX as usize) / (n_tracks.max(1) * n_bins.max(1));
        return Err(PyRuntimeError::new_err(format!(
            "batch_size={batch_size} causes Arrow i32 offset overflow. \
             Reduce batch_size to <= {safe_batch}."
        )));
    }

    // Open all BigWig handles once.
    let mut handles: Vec<BwHandle> = py
        .allow_threads(|| open_bigwig_handles(&bw_paths))
        .map_err(PyRuntimeError::new_err)?;

    let mut prof_chrom_scan_s: Vec<f64> = Vec::new();
    let valid_chroms: Vec<bool> = chrom_order
        .iter()
        .map(|(chrom, chrom_len_bp)| {
            samples_by_chrom.contains_key(chrom)
                && ((*chrom_len_bp as usize) / (bin_size as usize)) > 0
        })
        .collect();
    let total_output_shards: usize = chrom_order
        .iter()
        .zip(valid_chroms.iter())
        .filter(|(_, valid)| **valid)
        .map(|((chrom, _), _)| samples_by_chrom[chrom].len().div_ceil(batch_size))
        .sum();
    let mut next_output_shard = 0usize;

    let mut prof_writer_wall_s: Vec<f64> = Vec::new();
    let mut prof_write = WriteShardProfile::default();

    let started = Instant::now();
    let mut last_log = started;
    let total_samples = sample_indices.len();
    let mut done_samples = 0usize;

    for (shard_idx, (chrom, chrom_len_bp)) in chrom_order.iter().enumerate() {
        py.check_signals()?;
        let samples = &samples_by_chrom[chrom];

        // Drop the last partial bin so the region width is an exact multiple of
        // bin_size — this keeps `bin_region_into` on its SIMD fast path and
        // produces bit-identical bin values to the per-sample direct-bigwig path.
        let n_chrom_bins = (*chrom_len_bp as usize) / (bin_size as usize);
        let region_end = (n_chrom_bins as u32).saturating_mul(bin_size);
        if n_chrom_bins == 0 {
            eprintln!(
                "[regulonado_rs] chrom_pass[{shard_idx}/{n_chroms_used}] skipping chrom={chrom} (len {chrom_len_bp} < bin_size)",
            );
            continue;
        }
        let chrom_shard_start = next_output_shard;
        let chrom_n_shards = samples.len().div_ceil(batch_size);
        next_output_shard += chrom_n_shards;

        eprintln!(
            "[regulonado_rs] chrom_pass[{shard_idx}/{n_chroms_used}] chrom={chrom} samples={} chrom_bins={} region_end={}bp output_shards={}",
            samples.len(), n_chrom_bins, region_end, chrom_n_shards,
        );

        // --- Phase 1: bin the whole chromosome for every BigWig, in parallel ---
        let t_scan = Instant::now();
        let mut chrom_signals: Vec<f32> = vec![0.0; n_tracks * n_chrom_bins];
        let chrom_str: &str = chrom.as_str();

        py.allow_threads(|| {
            chrom_signals
                .par_chunks_mut(n_chrom_bins)
                .zip(handles.par_iter_mut())
                .enumerate()
                .for_each(|(track_idx, (out_row, reader))| {
                    let mut sums = vec![0.0f64; n_chrom_bins];
                    let mut covered = vec![0u64; n_chrom_bins];
                    let mut values_buf: Vec<f32> = Vec::new();

                    bin_region_into(
                        reader,
                        chrom_str,
                        0,
                        region_end,
                        out_row,
                        &mut sums,
                        &mut covered,
                        &mut values_buf,
                    );

                    // Minus-strand correction: if 80%+ of non-zero values are
                    // negative, negate the whole track. Mirrors
                    // bigwig_io::extract_bigwig_labels_batch so numerical output
                    // matches the existing direct-bigwig writer.
                    let is_minus = minus_flags.get(track_idx).copied().unwrap_or(false);
                    if is_minus {
                        let mut nz = 0usize;
                        let mut neg = 0usize;
                        for &v in out_row.iter() {
                            if v != 0.0 {
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
                });
        });
        let scan_s = t_scan.elapsed().as_secs_f64();
        if profile {
            prof_chrom_scan_s.push(scan_s);
        }

        // --- Phase 2: write row-block Arrow shards for this chrom in parallel ---
        let t_writer = Instant::now();
        let write_specs: Vec<WriteShardSpec<'_>> = (0..samples.len())
            .step_by(batch_size)
            .enumerate()
            .map(|(block_idx, batch_start)| WriteShardSpec {
                out_dir: &out_dir,
                shard_idx: chrom_shard_start + block_idx,
                shard_total: total_output_shards,
                batch_start,
                batch_end: (batch_start + batch_size).min(samples.len()),
                samples,
            })
            .collect();
        let write_ctx = WriteShardCtx {
            schema: Arc::clone(&schema),
            compression: &compression,
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
                "chrom_pass samples",
                done_samples,
                total_samples,
            );
        }

        if profile {
            prof_writer_wall_s.push(writer_wall_s);
        }
        // chrom_signals dropped here, freeing the bin matrix before the next chrom.
    }

    if profile {
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        eprintln!(
            "[regulonado_rs] chrom_pass profile chroms={n_chroms_used}\n  \
             chrom_scan: total_wall_s={:.1}\n  \
             writer:     total_wall_s={:.1} summed_worker_s={:.1} threads={}\n  \
             slice:      total_worker_s={:.1}\n  \
             fasta:      total_worker_s={:.1}\n  \
             batch:      total_worker_s={:.1}\n  \
             arrow_write: total_worker_s={:.1}\n  \
             bytes:      {:.1} GiB\n  \
             throughput: {:.2} samples/s\n  \
             total_wall_s={:.1}",
            sum(&prof_chrom_scan_s),
            sum(&prof_writer_wall_s),
            prof_write.wall_ns as f64 / 1e9,
            write_threads,
            prof_write.slice_ns as f64 / 1e9,
            prof_write.fasta_ns as f64 / 1e9,
            prof_write.batch_ns as f64 / 1e9,
            prof_write.arrow_write_ns as f64 / 1e9,
            prof_write.bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            prof_write.rows as f64 / started.elapsed().as_secs_f64().max(1e-9),
            started.elapsed().as_secs_f64(),
        );
    }

    Ok(())
}

struct SplitChromSamples {
    name: String,
    out_dir: String,
    total_samples: usize,
    samples_by_chrom: HashMap<String, Vec<(usize, usize)>>,
    total_shards: usize,
    next_shard: usize,
}

fn validate_batch_size(
    batch_size: usize,
    n_tracks: usize,
    n_bins: usize,
    context_len: usize,
) -> Result<(), String> {
    let max_label_offset = batch_size.saturating_mul(n_tracks).saturating_mul(n_bins);
    let max_input_offset = batch_size.saturating_mul(4).saturating_mul(context_len);
    if max_label_offset > i32::MAX as usize || max_input_offset > i32::MAX as usize {
        let safe_batch = (i32::MAX as usize) / (n_tracks.max(1) * n_bins.max(1));
        return Err(format!(
            "batch_size={batch_size} causes Arrow i32 offset overflow. \
             Reduce batch_size to <= {safe_batch}."
        ));
    }
    Ok(())
}

fn build_split_chrom_samples(
    split_names: Vec<String>,
    out_dirs: Vec<String>,
    split_sample_indices: Vec<Vec<usize>>,
    bed_rows: &[(String, u32, u32, String)],
    chrom_lengths: &HashMap<String, crate::fasta::FastaIndexRecord>,
    bin_size: u32,
    batch_size: usize,
) -> Result<(Vec<SplitChromSamples>, Vec<(String, u64)>), String> {
    if split_names.len() != out_dirs.len() || split_names.len() != split_sample_indices.len() {
        return Err(format!(
            "split_names, out_dirs, and split_sample_indices must have the same length \
             (got {}, {}, {})",
            split_names.len(),
            out_dirs.len(),
            split_sample_indices.len(),
        ));
    }

    let mut chrom_seen: HashMap<String, u64> = HashMap::new();
    let mut splits = Vec::with_capacity(split_names.len());
    for ((name, out_dir), sample_indices) in split_names
        .into_iter()
        .zip(out_dirs.into_iter())
        .zip(split_sample_indices.into_iter())
    {
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
            .map(|(_, samples)| samples.len().div_ceil(batch_size))
            .sum();

        splits.push(SplitChromSamples {
            name,
            out_dir,
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

#[pyfunction]
#[pyo3(signature = (
    bw_paths,
    minus_flags,
    signal_intervals,
    split_names,
    out_dirs,
    split_sample_indices,
    bed_rows,
    fasta_path,
    n_bins,
    context_len,
    bin_size,
    batch_size=4,
    n_threads=None,
    arrow_write_threads=None,
    compression=None,
    profile=false
))]
pub(crate) fn write_arrow_splits_chrom_pass(
    py: Python<'_>,
    bw_paths: Vec<String>,
    minus_flags: Vec<bool>,
    signal_intervals: Vec<(String, u32, u32)>,
    split_names: Vec<String>,
    out_dirs: Vec<String>,
    split_sample_indices: Vec<Vec<usize>>,
    bed_rows: Vec<(String, u32, u32, String)>,
    fasta_path: String,
    n_bins: usize,
    context_len: usize,
    bin_size: u32,
    batch_size: usize,
    n_threads: Option<usize>,
    arrow_write_threads: Option<usize>,
    compression: Option<String>,
    profile: bool,
) -> PyResult<()> {
    if let Some(nt) = n_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build_global()
            .ok();
    }

    let n_tracks = bw_paths.len();
    let batch_size = batch_size.max(1);
    validate_batch_size(batch_size, n_tracks, n_bins, context_len)
        .map_err(PyRuntimeError::new_err)?;

    let schema = hf_arrow_schema(context_len, n_tracks, n_bins);
    let fai = load_fasta_index(&fasta_path).map_err(PyRuntimeError::new_err)?;
    let (mut splits, chrom_order) = build_split_chrom_samples(
        split_names,
        out_dirs,
        split_sample_indices,
        &bed_rows,
        &fai,
        bin_size,
        batch_size,
    )
    .map_err(PyRuntimeError::new_err)?;

    let compression = compression.unwrap_or_else(|| "zstd".to_string());
    let write_threads = arrow_write_threads
        .unwrap_or_else(|| n_threads.unwrap_or(8).min(8))
        .max(1);
    let write_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(write_threads)
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("Cannot build Arrow write pool: {e}")))?;

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
        "[regulonado_rs] chrom_pass(all_splits): {} tracks × {} chromosomes ({} samples, {} shards) → [{}] (compression={}, arrow_write_threads={})",
        n_tracks,
        chrom_order.len(),
        total_samples,
        total_shards,
        split_summary,
        compression,
        write_threads,
    );

    let mut handles: Vec<BwHandle> = py
        .allow_threads(|| open_bigwig_handles(&bw_paths))
        .map_err(PyRuntimeError::new_err)?;

    let mut prof_chrom_scan_s: Vec<f64> = Vec::new();
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
            .map(|samples| samples.len().div_ceil(batch_size))
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

        py.allow_threads(|| {
            chrom_signals
                .par_chunks_mut(n_chrom_bins)
                .zip(handles.par_iter_mut())
                .enumerate()
                .for_each(|(track_idx, (out_row, reader))| {
                    let mut sums = vec![0.0f64; n_chrom_bins];
                    let mut covered = vec![0u64; n_chrom_bins];
                    let mut values_buf: Vec<f32> = Vec::new();

                    bin_region_into(
                        reader,
                        chrom_str,
                        0,
                        region_end,
                        out_row,
                        &mut sums,
                        &mut covered,
                        &mut values_buf,
                    );

                    let is_minus = minus_flags.get(track_idx).copied().unwrap_or(false);
                    if is_minus {
                        let mut nz = 0usize;
                        let mut neg = 0usize;
                        for &v in out_row.iter() {
                            if v != 0.0 {
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
                });
        });
        if profile {
            prof_chrom_scan_s.push(t_scan.elapsed().as_secs_f64());
        }

        let t_writer = Instant::now();
        let mut write_specs: Vec<WriteShardSpec<'_>> = Vec::new();
        for split in splits.iter_mut() {
            let Some(samples) = split.samples_by_chrom.get(chrom) else {
                continue;
            };
            for batch_start in (0..samples.len()).step_by(batch_size) {
                write_specs.push(WriteShardSpec {
                    out_dir: &split.out_dir,
                    shard_idx: split.next_shard,
                    shard_total: split.total_shards,
                    batch_start,
                    batch_end: (batch_start + batch_size).min(samples.len()),
                    samples,
                });
                split.next_shard += 1;
            }
        }

        let write_ctx = WriteShardCtx {
            schema: Arc::clone(&schema),
            compression: &compression,
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
             writer:     total_wall_s={:.1} summed_worker_s={:.1} threads={}\n  \
             slice:      total_worker_s={:.1}\n  \
             fasta:      total_worker_s={:.1}\n  \
             batch:      total_worker_s={:.1}\n  \
             arrow_write: total_worker_s={:.1}\n  \
             bytes:      {:.1} GiB\n  \
             throughput: {:.2} samples/s\n  \
             total_wall_s={:.1}",
            chrom_order.len(),
            sum(&prof_chrom_scan_s),
            sum(&prof_writer_wall_s),
            prof_write.wall_ns as f64 / 1e9,
            write_threads,
            prof_write.slice_ns as f64 / 1e9,
            prof_write.fasta_ns as f64 / 1e9,
            prof_write.batch_ns as f64 / 1e9,
            prof_write.arrow_write_ns as f64 / 1e9,
            prof_write.bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            prof_write.rows as f64 / started.elapsed().as_secs_f64().max(1e-9),
            started.elapsed().as_secs_f64(),
        );
    }

    for split in &splits {
        if split.next_shard != split.total_shards {
            return Err(PyRuntimeError::new_err(format!(
                "Split '{}' wrote {} shard(s), expected {}",
                split.name, split.next_shard, split.total_shards
            )));
        }
    }

    Ok(())
}
