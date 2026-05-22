use crate::binning::bin_region;
use crate::io_utils::maybe_log_progress;
use crate::signal_file::{process_one_track, process_one_track_at_offset};
use bigtools::BigWigRead;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Debug helper: extract one BigWig into one raw row-major float32 file.
///
/// Output is a raw little-endian float32 binary file with shape
/// `(len(intervals), n_bins)` in row-major order.  Read back in Python with::
///
///     arr = np.memmap(out_path, dtype="<f4", mode="r", shape=(n, n_bins))
///
/// Parameters
/// ----------
/// bw_path : str
///     Path to the BigWig file.
/// intervals : list[(str, int, int)]
///     ``(chrom, start, end)`` tuples (absolute genomic coordinates).
/// n_bins : int
///     Number of equal-width bins per interval.
/// out_path : str
///     Destination file for raw float32 output.
/// is_minus : bool, optional
///     If True AND 80 %+ of non-zero values are negative, negate all values.
#[pyfunction]
#[pyo3(signature = (bw_path, intervals, n_bins, out_path, is_minus=false))]
pub(crate) fn extract_bigwig_to_file(
    bw_path: String,
    intervals: Vec<(String, u32, u32)>,
    n_bins: usize,
    out_path: String,
    is_minus: bool,
) -> PyResult<()> {
    process_one_track(&bw_path, &intervals, n_bins, &out_path, is_minus)
        .map_err(PyRuntimeError::new_err)
}

/// Debug helper: extract one BigWig and return a NumPy array.
///
/// Returns
/// -------
/// np.ndarray, shape ``(n_intervals, n_bins)``, dtype float32
#[pyfunction]
#[pyo3(signature = (bw_path, intervals, n_bins, is_minus=false))]
pub(crate) fn extract_bigwig_regions<'py>(
    py: Python<'py>,
    bw_path: String,
    intervals: Vec<(String, u32, u32)>,
    n_bins: usize,
    is_minus: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut reader = BigWigRead::open_file(&bw_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Cannot open {bw_path}: {e}")))?;

    let n_samples = intervals.len();
    let mut data = Array2::<f32>::zeros((n_samples, n_bins));

    for (i, (chrom, start, end)) in intervals.iter().enumerate() {
        let bins = bin_region(&mut reader, chrom, *start, *end, n_bins);
        data.row_mut(i).assign(&ndarray::Array1::from_vec(bins));
    }

    if is_minus {
        let non_zero: Vec<f32> = data.iter().copied().filter(|v| *v != 0.0).collect();
        if !non_zero.is_empty() {
            let neg_frac =
                non_zero.iter().filter(|&&v| v < 0.0).count() as f32 / non_zero.len() as f32;
            if neg_frac >= 0.8 {
                data.mapv_inplace(|v| -v);
            }
        }
    }

    Ok(data.into_pyarray(py))
}

/// Compatibility helper: extract all tracks to one file per track.
///
/// Each output file is a raw float32 array of shape ``(n_intervals, n_bins)``.
/// This is useful for older benchmarks, but the production path uses
/// `extract_all_tracks_to_file` so Arrow writing can read one shared
/// track-major signal file.
///
/// Ctrl+C / KeyboardInterrupt is checked between chunks of tracks: the GIL is
/// released for each chunk (so Rayon threads run freely), then re-acquired to
/// call `check_signals()` before starting the next chunk.
///
/// Parameters
/// ----------
/// bw_paths : list[str]
///     Ordered list of BigWig paths.
/// intervals : list[(str, int, int)]
///     ``(chrom, start, end)`` tuples.
/// n_bins : int
///     Bins per interval.
/// out_dir : str
///     Directory for output files.  Each track writes ``<out_dir>/track_<i>.bin``.
/// minus_flags : list[bool], optional
///     Per-track is_minus flag (same length as bw_paths).
/// n_threads : int, optional
///     Rayon thread count (default: all CPUs).
#[pyfunction]
#[pyo3(signature = (bw_paths, intervals, n_bins, out_dir, minus_flags=None, n_threads=None))]
pub(crate) fn extract_all_tracks_to_dir(
    py: Python<'_>,
    bw_paths: Vec<String>,
    intervals: Vec<(String, u32, u32)>,
    n_bins: usize,
    out_dir: String,
    minus_flags: Option<Vec<bool>>,
    n_threads: Option<usize>,
) -> PyResult<()> {
    let n_threads = n_threads.unwrap_or(32);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok(); // no-op if pool already built

    let out_dir_path = Path::new(&out_dir);
    let intervals = Arc::new(intervals);
    let flags = minus_flags.unwrap_or_else(|| vec![false; bw_paths.len()]);

    // Process tracks in chunks of n_threads so all Rayon threads stay saturated
    // within each chunk. Between chunks we hold the GIL and call check_signals()
    // so Ctrl+C is noticed promptly (at most one chunk's worth of delay).
    let mut all_errors: Vec<String> = Vec::new();

    for chunk_start in (0..bw_paths.len()).step_by(n_threads) {
        // Check for Ctrl+C / KeyboardInterrupt before each chunk.
        // check_signals() only works on the main Python thread, which is here.
        py.check_signals()?;

        let chunk_end = (chunk_start + n_threads).min(bw_paths.len());
        let intervals_ref = Arc::clone(&intervals);

        let chunk_errors: Vec<String> = py.allow_threads(|| {
            (chunk_start..chunk_end)
                .into_par_iter()
                .filter_map(|idx| {
                    let out_path = out_dir_path
                        .join(format!("track_{idx:06}.bin"))
                        .to_string_lossy()
                        .into_owned();
                    let is_minus = flags.get(idx).copied().unwrap_or(false);
                    process_one_track(&bw_paths[idx], &intervals_ref, n_bins, &out_path, is_minus)
                        .err()
                })
                .collect()
        });

        all_errors.extend(chunk_errors);
    }

    if all_errors.is_empty() {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err(format!(
            "{} track(s) failed:\n{}",
            all_errors.len(),
            all_errors.join("\n")
        )))
    }
}

/// Production signal extractor: write all tracks to one track-major raw file.
///
/// Output shape is ``(n_tracks, n_intervals, n_bins)`` in row-major order.
/// Python can open it with::
///
///     arr = np.memmap(path, dtype="<f4", mode="r", shape=(t, n, b))
///
/// Each Rayon worker owns a track and writes it into a fixed byte range. This
/// keeps the fast BigWig access pattern (open one track, read all intervals)
/// while producing the layout expected by `write_arrow_split_from_track_major`.
#[pyfunction]
#[pyo3(signature = (bw_paths, intervals, n_bins, out_path, minus_flags=None, n_threads=None))]
pub(crate) fn extract_all_tracks_to_file(
    py: Python<'_>,
    bw_paths: Vec<String>,
    intervals: Vec<(String, u32, u32)>,
    n_bins: usize,
    out_path: String,
    minus_flags: Option<Vec<bool>>,
    n_threads: Option<usize>,
) -> PyResult<()> {
    let n_threads = n_threads.unwrap_or(32);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok(); // no-op if pool already built

    let row_bytes = n_bins
        .checked_mul(4)
        .ok_or_else(|| PyRuntimeError::new_err("n_bins is too large"))?;
    let track_bytes = intervals
        .len()
        .checked_mul(row_bytes)
        .ok_or_else(|| PyRuntimeError::new_err("output file would be too large"))?;
    let total_bytes = bw_paths
        .len()
        .checked_mul(track_bytes)
        .ok_or_else(|| PyRuntimeError::new_err("output file would be too large"))?;

    let file = std::fs::File::create(&out_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Cannot create {out_path}: {e}")))?;
    file.set_len(total_bytes as u64)
        .map_err(|e| PyRuntimeError::new_err(format!("Cannot size {out_path}: {e}")))?;
    drop(file);

    let intervals = Arc::new(intervals);
    let flags = minus_flags.unwrap_or_else(|| vec![false; bw_paths.len()]);
    let mut all_errors: Vec<String> = Vec::new();
    let started = Instant::now();
    let mut last_log = started;

    eprintln!(
        "[regulonado_rs] extract(track-major): writing {} tracks x {} intervals x {} bins to {}",
        bw_paths.len(),
        intervals.len(),
        n_bins,
        out_path,
    );

    for chunk_start in (0..bw_paths.len()).step_by(n_threads) {
        py.check_signals()?;

        let chunk_end = (chunk_start + n_threads).min(bw_paths.len());
        let intervals_ref = Arc::clone(&intervals);
        let out_path_ref = &out_path;

        let chunk_errors: Vec<String> = py.allow_threads(|| {
            (chunk_start..chunk_end)
                .into_par_iter()
                .filter_map(|idx| {
                    let offset = (idx * track_bytes) as u64;
                    let is_minus = flags.get(idx).copied().unwrap_or(false);
                    process_one_track_at_offset(
                        &bw_paths[idx],
                        &intervals_ref,
                        n_bins,
                        out_path_ref,
                        offset,
                        track_bytes as u64,
                        is_minus,
                    )
                    .err()
                })
                .collect()
        });

        all_errors.extend(chunk_errors);
        maybe_log_progress(
            &mut last_log,
            started,
            "extract(track-major) tracks",
            chunk_end,
            bw_paths.len(),
        );
    }

    if all_errors.is_empty() {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err(format!(
            "{} track(s) failed:\n{}",
            all_errors.len(),
            all_errors.join("\n")
        )))
    }
}
