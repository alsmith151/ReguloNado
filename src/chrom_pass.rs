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
    ArrayRef, RecordBatch,
    builder::{Int64Builder, StringBuilder},
};
use arrow_ipc::writer::StreamWriter;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::bigwig_io::{BwHandle, open_bigwig_handles};
use crate::binning::bin_region_into;
use crate::fasta::{load_fasta_index, read_one_hot_sequence};
use crate::io_utils::{ipc_write_options, maybe_log_progress};
use crate::schema::{hf_arrow_schema, make_2d_f32_array, make_2d_i8_array};

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
    compression: Option<String>,
    profile: bool,
) -> PyResult<()> {
    if let Some(nt) = n_threads {
        rayon::ThreadPoolBuilder::new().num_threads(nt).build_global().ok();
    }

    let n_tracks = bw_paths.len();
    let schema = hf_arrow_schema(context_len, n_tracks, n_bins);

    let fai = load_fasta_index(&fasta_path).map_err(PyRuntimeError::new_err)?;
    let fasta = std::fs::File::open(&fasta_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Cannot open FASTA {fasta_path}: {e}")))?;

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
    chrom_order.sort_by(|a, b| b.1.cmp(&a.1));

    let n_chroms_used = chrom_order.len();
    let compression = compression.unwrap_or_else(|| "zstd".to_string());

    eprintln!(
        "[regulonado_rs] chrom_pass: {} tracks × {} chromosomes ({} samples) → {} (compression={})",
        n_tracks,
        n_chroms_used,
        sample_indices.len(),
        out_dir,
        compression,
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
    let mut prof_slice_s: Vec<f64> = Vec::new();
    let mut prof_fasta_s: Vec<f64> = Vec::new();
    let mut prof_arrow_s: Vec<f64> = Vec::new();

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

        eprintln!(
            "[regulonado_rs] chrom_pass[{shard_idx}/{n_chroms_used}] chrom={chrom} samples={} chrom_bins={} region_end={}bp",
            samples.len(), n_chrom_bins, region_end,
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

        // --- Phase 2: write samples on this chrom into a per-chrom Arrow shard ---
        let shard_path = format!(
            "{}/data-{:05}-of-{:05}.arrow",
            out_dir, shard_idx, n_chroms_used
        );
        let out_file = std::fs::File::create(&shard_path).map_err(|e| {
            PyRuntimeError::new_err(format!("Cannot create Arrow file {shard_path}: {e}"))
        })?;
        let write_options = ipc_write_options(&compression).map_err(PyRuntimeError::new_err)?;
        let mut writer = StreamWriter::try_new_with_options(out_file, &schema, write_options)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut slice_ns: u128 = 0;
        let mut fasta_ns: u128 = 0;
        let mut arrow_ns: u128 = 0;

        for batch_start in (0..samples.len()).step_by(batch_size) {
            py.check_signals()?;
            let batch_end = (batch_start + batch_size).min(samples.len());
            let rows_in_batch = batch_end - batch_start;

            // Slice labels from chrom_signals into a contiguous (rows, tracks, bins) buffer.
            let t_slice = Instant::now();
            let mut labels: Vec<f32> = vec![0.0; rows_in_batch * n_tracks * n_bins];
            for (row_idx, (_, global_idx)) in
                samples[batch_start..batch_end].iter().copied().enumerate()
            {
                let (_chrom, sig_start, sig_end) = &signal_intervals[global_idx];
                let bin_start = (*sig_start / bin_size) as usize;
                let bin_end_raw = (*sig_end / bin_size) as usize;
                let bin_end = bin_end_raw.min(n_chrom_bins);
                let copy_n = bin_end.saturating_sub(bin_start).min(n_bins);

                if copy_n > 0 && bin_start < n_chrom_bins {
                    let row_base = row_idx * n_tracks * n_bins;
                    for track_idx in 0..n_tracks {
                        let src_start = track_idx * n_chrom_bins + bin_start;
                        let dst_start = row_base + track_idx * n_bins;
                        labels[dst_start..dst_start + copy_n]
                            .copy_from_slice(&chrom_signals[src_start..src_start + copy_n]);
                        // Trailing bins remain zero from vec init.
                    }
                }
            }
            if profile {
                slice_ns += t_slice.elapsed().as_nanos();
            }

            // Build DNA + metadata columns.
            let t_fasta = Instant::now();
            let mut input_values = Vec::with_capacity(rows_in_batch * 4 * context_len);
            let mut interval_builder =
                StringBuilder::with_capacity(rows_in_batch, rows_in_batch * 32);
            let mut index_builder = Int64Builder::with_capacity(rows_in_batch);
            let mut local_index_builder = Int64Builder::with_capacity(rows_in_batch);

            for (local_idx, global_idx) in samples[batch_start..batch_end].iter().copied() {
                let (bed_chrom, bed_start, bed_end, _) = &bed_rows[global_idx];
                let seq = read_one_hot_sequence(
                    &fasta,
                    &fai,
                    bed_chrom,
                    *bed_start,
                    *bed_end,
                    context_len,
                )
                .map_err(PyRuntimeError::new_err)?;
                input_values.extend_from_slice(&seq);
                interval_builder.append_value(format!("{bed_chrom}:{bed_start}-{bed_end}"));
                index_builder.append_value(global_idx as i64);
                local_index_builder.append_value(local_idx as i64);
            }
            if profile {
                fasta_ns += t_fasta.elapsed().as_nanos();
            }

            // Write the Arrow record batch.
            let t_arrow = Instant::now();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    make_2d_i8_array(input_values, rows_in_batch, 4, context_len),
                    make_2d_f32_array(labels, rows_in_batch, n_tracks, n_bins),
                    Arc::new(interval_builder.finish()) as ArrayRef,
                    Arc::new(index_builder.finish()) as ArrayRef,
                    Arc::new(local_index_builder.finish()) as ArrayRef,
                ],
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            writer
                .write(&batch)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if profile {
                arrow_ns += t_arrow.elapsed().as_nanos();
            }

            done_samples += rows_in_batch;
            maybe_log_progress(
                &mut last_log,
                started,
                "chrom_pass samples",
                done_samples,
                total_samples,
            );
        }

        writer
            .finish()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if profile {
            prof_slice_s.push(slice_ns as f64 / 1e9);
            prof_fasta_s.push(fasta_ns as f64 / 1e9);
            prof_arrow_s.push(arrow_ns as f64 / 1e9);
        }
        // chrom_signals dropped here, freeing the bin matrix before the next chrom.
    }

    if profile {
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        eprintln!(
            "[regulonado_rs] chrom_pass profile chroms={n_chroms_used}\n  \
             chrom_scan: total_wall_s={:.1}\n  \
             slice:      total_wall_s={:.1}\n  \
             fasta:      total_wall_s={:.1}\n  \
             arrow:      total_wall_s={:.1}\n  \
             total_wall_s={:.1}",
            sum(&prof_chrom_scan_s),
            sum(&prof_slice_s),
            sum(&prof_fasta_s),
            sum(&prof_arrow_s),
            started.elapsed().as_secs_f64(),
        );
    }

    Ok(())
}
