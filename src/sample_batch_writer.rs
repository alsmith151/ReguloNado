use crate::arrow_schema::{sequence_tokens_array, signal_array, window_arrow_schema};
use crate::bigwig_io::{extract_bigwig_labels_batch, open_bigwig_handles};
use crate::binning::BinningOptions;
use crate::fasta::{load_fasta_index, read_sequence_tokens};
use crate::io_utils::{maybe_log_progress, parquet_writer_properties};
use arrow_array::{
    builder::{Int64Builder, StringBuilder},
    ArrayRef, RecordBatch,
};
use parquet::arrow::ArrowWriter;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;
use std::time::Instant;

/// Production low-scratch Parquet writer: emit one split directly from BigWigs.
///
/// This writer avoids the dense `(n_tracks, n_intervals, n_bins)` raw scratch
/// buffer used by the track-major benchmark path. For each sample it reads the
/// selected signal interval from every BigWig, reads the matching sequence
/// tokens from FASTA, and appends one row to a Parquet file (kept as a parity
/// reference for `chromosome_scan_writer`; both share the same schema and
/// writer settings).
#[pyfunction]
#[pyo3(signature = (
    bw_paths,
    minus_flags,
    signal_intervals,
    parquet_path,
    sample_indices,
    bed_rows,
    fasta_path,
    n_bins,
    context_len,
    hf_features_json,
    rows_per_row_group=1,
    zstd_level=3,
    n_threads=None,
    mean_over_covered_bases=false,
    missing_as_nan=true
))]
// Arity is dictated by the Python-facing signature above, so grouping the
// parameters into a struct would only move the problem to the call site.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_parquet_split_from_bigwigs(
    py: Python<'_>,
    bw_paths: Vec<String>,
    minus_flags: Vec<bool>,
    signal_intervals: Vec<(String, u32, u32)>,
    parquet_path: String,
    sample_indices: Vec<usize>,
    bed_rows: Vec<(String, u32, u32, String)>,
    fasta_path: String,
    n_bins: usize,
    context_len: usize,
    hf_features_json: String,
    rows_per_row_group: usize,
    zstd_level: i32,
    n_threads: Option<usize>,
    mean_over_covered_bases: bool,
    missing_as_nan: bool,
) -> PyResult<usize> {
    let binning_options = BinningOptions {
        mean_over_covered_bases,
        missing_as_nan,
    };
    if let Some(n_threads) = n_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .ok();
    }

    let n_tracks = bw_paths.len();
    let schema = window_arrow_schema(context_len, n_tracks, n_bins, hf_features_json);
    let fasta = std::fs::File::open(&fasta_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Cannot open FASTA {fasta_path}: {e}")))?;
    let fai = load_fasta_index(&fasta_path).map_err(PyRuntimeError::new_err)?;
    let out_file = std::fs::File::create(&parquet_path).map_err(|e| {
        PyRuntimeError::new_err(format!("Cannot create Parquet file {parquet_path}: {e}"))
    })?;
    let props = parquet_writer_properties(rows_per_row_group, zstd_level)
        .map_err(PyRuntimeError::new_err)?;
    let mut writer = ArrowWriter::try_new(out_file, Arc::clone(&schema), Some(props))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let total_rows = sample_indices.len();
    let started = Instant::now();
    let mut last_log = started;
    eprintln!(
        "[regulonado_rs] parquet(direct-bigwig): writing {} samples x {} tracks to {}",
        total_rows, n_tracks, parquet_path,
    );

    let mut bw_handles = py
        .allow_threads(|| open_bigwig_handles(&bw_paths))
        .map_err(PyRuntimeError::new_err)?;

    for (row_offset, global_idx) in sample_indices.iter().copied().enumerate() {
        py.check_signals()?;

        let interval = signal_intervals.get(global_idx).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "Signal interval index {global_idx} is out of range"
            ))
        })?;

        let labels = py
            .allow_threads(|| {
                extract_bigwig_labels_batch(
                    &mut bw_handles,
                    &minus_flags,
                    std::slice::from_ref(interval),
                    n_bins,
                    binning_options,
                )
            })
            .map_err(PyRuntimeError::new_err)?;

        let (chrom, start, end, _) = bed_rows.get(global_idx).ok_or_else(|| {
            PyRuntimeError::new_err(format!("BED index {global_idx} is out of range"))
        })?;
        let tokens = read_sequence_tokens(&fasta, &fai, chrom, *start, *end, context_len)
            .map_err(PyRuntimeError::new_err)?;

        let mut interval_builder = StringBuilder::with_capacity(1, 32);
        let mut index_builder = Int64Builder::with_capacity(1);
        let mut local_index_builder = Int64Builder::with_capacity(1);
        interval_builder.append_value(format!("{chrom}:{start}-{end}"));
        index_builder.append_value(global_idx as i64);
        local_index_builder.append_value(row_offset as i64);

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                sequence_tokens_array(tokens, context_len),
                signal_array(labels, n_tracks, n_bins),
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
            "parquet(direct-bigwig) rows",
            row_offset + 1,
            total_rows,
        );
    }

    writer
        .close()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(total_rows)
}
