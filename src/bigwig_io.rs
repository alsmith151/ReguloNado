use bigtools::{BigWigRead, utils::file::reopen::ReopenableFile};
use rayon::prelude::*;
use std::time::Instant;
use crate::binning::bin_region_into;

pub(crate) type BwHandle = BigWigRead<ReopenableFile>;

/// Open all BigWig handles in parallel.
///
/// Returns the handles plus the summed thread-time spent in open calls (nanoseconds).
/// Caller should invoke this once before the batch loop and reuse the handles.
pub(crate) fn open_bigwig_handles(bw_paths: &[String], profile: bool) -> Result<(Vec<BwHandle>, u64), String> {
    let results: Vec<Result<(BwHandle, u64), String>> = bw_paths
        .par_iter()
        .map(|p| {
            let t = if profile { Some(Instant::now()) } else { None };
            let h = BigWigRead::open_file(p).map_err(|e| format!("Cannot open {p}: {e}"))?;
            let ns = t.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);
            Ok((h, ns))
        })
        .collect();
    let mut handles = Vec::with_capacity(bw_paths.len());
    let mut total_ns: u64 = 0;
    for r in results {
        let (h, ns) = r?;
        handles.push(h);
        total_ns += ns;
    }
    Ok((handles, total_ns))
}

/// Extract one Arrow record batch worth of labels directly from BigWigs.
///
/// Output layout is `(rows, n_tracks, n_bins)`, matching the nested Arrow label
/// field. Handles must be opened once by the caller (via `open_bigwig_handles`)
/// and reused across batches.
/// Returned alongside the label buffer when `profile=true`.
/// All values are nanoseconds; `read_bin` is summed thread-time
/// (may exceed wall time on a multi-core Rayon pool).
pub(crate) struct BatchTimings {
    pub(crate) opens_thread_ns: u64,
    pub(crate) read_bin_thread_ns: u64,
    pub(crate) assemble_wall_ns: u64,
}

pub(crate) fn extract_bigwig_labels_batch(
    handles: &mut [BwHandle],
    minus_flags: &[bool],
    intervals: &[(String, u32, u32)],
    n_bins: usize,
    profile: bool,
) -> Result<(Vec<f32>, BatchTimings), String> {
    let rows = intervals.len();
    let n_tracks = handles.len();
    let row_values = n_tracks
        .checked_mul(n_bins)
        .ok_or_else(|| "label row shape is too large".to_string())?;
    let mut labels = vec![0.0f32; rows * row_values];

    let track_results: Vec<Result<(usize, Vec<f32>, u64), String>> = handles
        .par_iter_mut()
        .enumerate()
        .map(|(track_idx, reader)| {
            let mut values = vec![0.0f32; rows * n_bins];
            let mut sums = vec![0.0f64; n_bins];
            let mut covered = vec![0u64; n_bins];
            let mut values_buf: Vec<f32> = Vec::new();
            let mut neg = 0usize;
            let mut nz = 0usize;
            let is_minus = minus_flags.get(track_idx).copied().unwrap_or(false);

            let t_read = if profile { Some(Instant::now()) } else { None };
            for (row_idx, (chrom, start, end)) in intervals.iter().enumerate() {
                let row_start = row_idx * n_bins;
                let row = &mut values[row_start..row_start + n_bins];
                bin_region_into(
                    reader,
                    chrom,
                    *start,
                    *end,
                    row,
                    &mut sums,
                    &mut covered,
                    &mut values_buf,
                );

                if is_minus {
                    for &v in row.iter() {
                        if v != 0.0 {
                            nz += 1;
                            if v < 0.0 {
                                neg += 1;
                            }
                        }
                    }
                }
            }
            let read_bin_ns = t_read.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);

            if is_minus && nz > 0 && neg as f32 / nz as f32 >= 0.8 {
                for v in &mut values {
                    *v = -*v;
                }
            }

            Ok((track_idx, values, read_bin_ns))
        })
        .collect();

    let mut total_read_bin_ns: u64 = 0;
    let t_assemble = if profile { Some(Instant::now()) } else { None };
    for result in track_results {
        let (track_idx, values, read_bin_ns) = result?;
        total_read_bin_ns += read_bin_ns;
        for row_idx in 0..rows {
            let src_start = row_idx * n_bins;
            let dst_start = (row_idx * n_tracks + track_idx) * n_bins;
            labels[dst_start..dst_start + n_bins]
                .copy_from_slice(&values[src_start..src_start + n_bins]);
        }
    }
    let assemble_wall_ns = t_assemble.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);

    Ok((labels, BatchTimings { opens_thread_ns: 0, read_bin_thread_ns: total_read_bin_ns, assemble_wall_ns }))
}
