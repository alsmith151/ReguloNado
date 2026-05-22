use bigtools::{BigWigRead, utils::file::reopen::ReopenableFile};
use rayon::prelude::*;
use crate::binning::bin_region_into;

pub(crate) type BwHandle = BigWigRead<ReopenableFile>;

/// Open all BigWig handles in parallel.
///
/// Caller should invoke this once before the batch loop and reuse the handles.
pub(crate) fn open_bigwig_handles(bw_paths: &[String]) -> Result<Vec<BwHandle>, String> {
    bw_paths
        .par_iter()
        .map(|p| BigWigRead::open_file(p).map_err(|e| format!("Cannot open {p}: {e}")))
        .collect()
}

/// Extract one Arrow record batch worth of labels directly from BigWigs.
///
/// Output layout is `(rows, n_tracks, n_bins)`, matching the nested Arrow label
/// field. Handles must be opened once by the caller (via `open_bigwig_handles`)
/// and reused across batches.
pub(crate) fn extract_bigwig_labels_batch(
    handles: &mut [BwHandle],
    minus_flags: &[bool],
    intervals: &[(String, u32, u32)],
    n_bins: usize,
) -> Result<Vec<f32>, String> {
    let rows = intervals.len();
    let n_tracks = handles.len();
    let row_values = n_tracks
        .checked_mul(n_bins)
        .ok_or_else(|| "label row shape is too large".to_string())?;
    let mut labels = vec![0.0f32; rows * row_values];

    let track_results: Vec<Result<(usize, Vec<f32>), String>> = handles
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

            if is_minus && nz > 0 && neg as f32 / nz as f32 >= 0.8 {
                for v in &mut values {
                    *v = -*v;
                }
            }

            Ok((track_idx, values))
        })
        .collect();

    for result in track_results {
        let (track_idx, values) = result?;
        for row_idx in 0..rows {
            let src_start = row_idx * n_bins;
            let dst_start = (row_idx * n_tracks + track_idx) * n_bins;
            labels[dst_start..dst_start + n_bins]
                .copy_from_slice(&values[src_start..src_start + n_bins]);
        }
    }

    Ok(labels)
}
