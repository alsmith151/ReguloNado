use crate::bigwig_io::{extract_bigwig_labels_batch, open_bigwig_handles};
use crate::fasta::{load_fasta_index, read_one_hot_sequence};
use crate::io_utils::{ipc_write_options, maybe_log_progress};
use crate::schema::{
    hf_arrow_schema, make_2d_f32_array, make_2d_i8_array,
};
use arrow_array::{
    builder::{Int64Builder, StringBuilder},
    ArrayRef, RecordBatch,
};
use arrow_ipc::writer::StreamWriter;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon;
use std::sync::Arc;
use std::time::Instant;

/// Production low-scratch Arrow writer: emit one split directly from BigWigs.
///
/// This writer avoids the dense `(n_tracks, n_intervals, n_bins)` raw scratch
/// file used by the track-major benchmark path. For each Arrow record batch it
/// reads the selected signal intervals from every BigWig, builds a dense label
/// buffer for that batch, reads matching one-hot sequence from FASTA, and writes
/// the batch to a HuggingFace-compatible Arrow IPC stream.
#[pyfunction]
#[pyo3(signature = (
    bw_paths,
    minus_flags,
    signal_intervals,
    arrow_path,
    sample_indices,
    bed_rows,
    fasta_path,
    n_bins,
    context_len,
    batch_size=4,
    n_threads=None,
    compression=None
))]
pub(crate) fn write_arrow_split_from_bigwigs(
    py: Python<'_>,
    bw_paths: Vec<String>,
    minus_flags: Vec<bool>,
    signal_intervals: Vec<(String, u32, u32)>,
    arrow_path: String,
    sample_indices: Vec<usize>,
    bed_rows: Vec<(String, u32, u32, String)>,
    fasta_path: String,
    n_bins: usize,
    context_len: usize,
    batch_size: usize,
    n_threads: Option<usize>,
    compression: Option<String>,
) -> PyResult<()> {
    if let Some(n_threads) = n_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .ok();
    }

    let n_tracks = bw_paths.len();
    let schema = hf_arrow_schema(context_len, n_tracks, n_bins);
    let fasta = std::fs::File::open(&fasta_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Cannot open FASTA {fasta_path}: {e}")))?;
    let fai = load_fasta_index(&fasta_path).map_err(PyRuntimeError::new_err)?;
    let out_file = std::fs::File::create(&arrow_path).map_err(|e| {
        PyRuntimeError::new_err(format!("Cannot create Arrow file {arrow_path}: {e}"))
    })?;
    let compression = compression.unwrap_or_else(|| "zstd".to_string());
    let write_options = ipc_write_options(&compression).map_err(PyRuntimeError::new_err)?;
    let mut writer = StreamWriter::try_new_with_options(out_file, &schema, write_options)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let batch_size = batch_size.max(1);

    // Arrow ListArray uses i32 offsets.  The inner-most offset for labels is
    // batch_size * n_tracks * n_bins; for inputs it is batch_size * 4 * context_len.
    // Both must fit in i32.
    let max_label_offset = batch_size.saturating_mul(n_tracks).saturating_mul(n_bins);
    let max_input_offset = batch_size.saturating_mul(4).saturating_mul(context_len);
    if max_label_offset > i32::MAX as usize || max_input_offset > i32::MAX as usize {
        let safe_batch = (i32::MAX as usize) / (n_tracks.max(1) * n_bins.max(1));
        return Err(PyRuntimeError::new_err(format!(
            "batch_size={batch_size} causes Arrow i32 offset overflow \
             (batch * n_tracks * n_bins = {max_label_offset} > i32::MAX). \
             Reduce --arrow-batch-size to <= {safe_batch}."
        )));
    }

    let total_batches = sample_indices.len().div_ceil(batch_size);
    let started = Instant::now();
    let mut last_log = started;
    eprintln!(
        "[regulonado_rs] arrow(direct-bigwig): writing {} samples x {} tracks in {} batches to {} (compression={})",
        sample_indices.len(),
        n_tracks,
        total_batches,
        arrow_path,
        compression,
    );

    let mut bw_handles = py
        .allow_threads(|| open_bigwig_handles(&bw_paths))
        .map_err(PyRuntimeError::new_err)?;

    for batch_start in (0..sample_indices.len()).step_by(batch_size) {
        py.check_signals()?;

        let batch_end = (batch_start + batch_size).min(sample_indices.len());
        let rows_in_batch = batch_end - batch_start;
        let batch_indices = &sample_indices[batch_start..batch_end];

        let mut batch_signal_intervals = Vec::with_capacity(rows_in_batch);
        for global_idx in batch_indices.iter().copied() {
            let interval = signal_intervals.get(global_idx).ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "Signal interval index {global_idx} is out of range"
                ))
            })?;
            batch_signal_intervals.push(interval.clone());
        }

        let labels = py
            .allow_threads(|| {
                extract_bigwig_labels_batch(
                    &mut bw_handles,
                    &minus_flags,
                    &batch_signal_intervals,
                    n_bins,
                )
            })
            .map_err(PyRuntimeError::new_err)?;

        let mut input_values = Vec::with_capacity(rows_in_batch * 4 * context_len);
        let mut interval_builder = StringBuilder::with_capacity(rows_in_batch, rows_in_batch * 32);
        let mut index_builder = Int64Builder::with_capacity(rows_in_batch);
        let mut local_index_builder = Int64Builder::with_capacity(rows_in_batch);

        for (row_offset, global_idx) in batch_indices.iter().copied().enumerate() {
            let (chrom, start, end, _) = bed_rows.get(global_idx).ok_or_else(|| {
                PyRuntimeError::new_err(format!("BED index {global_idx} is out of range"))
            })?;
            let seq = read_one_hot_sequence(&fasta, &fai, chrom, *start, *end, context_len)
                .map_err(PyRuntimeError::new_err)?;

            input_values.extend_from_slice(&seq);
            interval_builder.append_value(format!("{chrom}:{start}-{end}"));
            index_builder.append_value(global_idx as i64);
            local_index_builder.append_value((batch_start + row_offset) as i64);
        }

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
        maybe_log_progress(
            &mut last_log,
            started,
            "arrow(direct-bigwig) batches",
            batch_start / batch_size + 1,
            total_batches,
        );
    }

    writer
        .finish()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}
