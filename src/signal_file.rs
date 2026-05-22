use crate::binning::bin_region_into;
use bigtools::{BBIFileRead, BigWigRead};
use std::io::{BufWriter, Write};

/// Negate all float32 values in a raw binary file in-place.
///
/// Negation is done by flipping the IEEE 754 sign bit (byte 3 on little-endian),
/// so no f32 parse/format round-trip is needed.
pub(crate) fn negate_file_inplace(path: &str) -> Result<(), String> {
    let len = std::fs::metadata(path)
        .map_err(|e| format!("Cannot stat {path} for in-place negation: {e}"))?
        .len();
    negate_file_range_inplace(path, 0, len)
}

pub(crate) fn negate_file_range_inplace(path: &str, offset: u64, len: u64) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| format!("Cannot open {path} for in-place negation: {e}"))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("Seek error: {e}"))?;

    let mut buf = vec![0u8; 4 * 1024 * 1024];
    let mut remaining = len;
    while remaining > 0 {
        let pos = file
            .stream_position()
            .map_err(|e| format!("Seek error: {e}"))?;
        let read_len = buf.len().min(remaining as usize);
        let n = file
            .read(&mut buf[..read_len])
            .map_err(|e| format!("Read error: {e}"))?;
        if n == 0 {
            break;
        }
        // Flip sign bit of each f32 (byte 3 is the MSB on LE / x86-64).
        for chunk in buf[..n].chunks_exact_mut(4) {
            chunk[3] ^= 0x80;
        }
        file.seek(SeekFrom::Start(pos))
            .map_err(|e| format!("Seek error: {e}"))?;
        file.write_all(&buf[..n])
            .map_err(|e| format!("Write error: {e}"))?;
        remaining -= n as u64;
    }
    Ok(())
}

/// Stream one track's binned signal rows to any writer.
///
/// The row and binning scratch buffers are reused across intervals. If
/// `is_minus` is true, this also accumulates non-zero and negative-value counts
/// so the caller can decide whether to sign-flip the written track.
pub(crate) fn write_track_rows<R: BBIFileRead, W: Write>(
    reader: &mut BigWigRead<R>,
    writer: &mut W,
    intervals: &[(String, u32, u32)],
    n_bins: usize,
    is_minus: bool,
) -> Result<(usize, usize), String> {
    // Single row reused across all intervals — peak allocation is 24 KB, not 1 GB.
    let mut row = vec![0.0f32; n_bins];
    let mut sums = vec![0.0f64; n_bins];
    let mut covered = vec![0u64; n_bins];
    let mut values_buf: Vec<f32> = Vec::new();

    // Minus-strand stats accumulated during write to avoid a second BigWig read.
    let mut neg = 0usize;
    let mut nz = 0usize;

    for (chrom, start, end) in intervals.iter() {
        bin_region_into(
            &mut *reader,
            chrom,
            *start,
            *end,
            &mut row,
            &mut sums,
            &mut covered,
            &mut values_buf,
        );

        if is_minus {
            for &v in &row {
                if v != 0.0 {
                    nz += 1;
                    if v < 0.0 {
                        neg += 1;
                    }
                }
            }
        }

        // Safe: x86-64 is little-endian, f32 memory layout == LE bytes.
        let bytes = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len() * 4) };
        writer
            .write_all(bytes)
            .map_err(|e| format!("Write error: {e}"))?;
    }

    Ok((neg, nz))
}

/// Open one BigWig, bin all intervals, and write a standalone track file.
///
/// Streams one row (n_bins floats = 24 KB) at a time rather than accumulating
/// the whole track in memory (~1 GB at 41 k intervals × 6144 bins).  Minus-strand
/// statistics are accumulated during the write pass; if correction is needed,
/// the file is negated in-place with a second streaming pass (sign-bit flip,
/// no extra heap allocation).
pub(crate) fn process_one_track(
    bw_path: &str,
    intervals: &[(String, u32, u32)],
    n_bins: usize,
    out_path: &str,
    is_minus: bool,
) -> Result<(), String> {
    let mut reader =
        BigWigRead::open_file(bw_path).map_err(|e| format!("Cannot open {bw_path}: {e}"))?;

    let file =
        std::fs::File::create(out_path).map_err(|e| format!("Cannot create {out_path}: {e}"))?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

    let (neg, nz) = write_track_rows(&mut reader, &mut writer, intervals, n_bins, is_minus)?;
    writer.flush().map_err(|e| format!("Flush error: {e}"))?;
    drop(writer); // Close before potential in-place rewrite.

    if is_minus && nz > 0 && neg as f32 / nz as f32 >= 0.8 {
        negate_file_inplace(out_path)?;
    }

    Ok(())
}

/// Write one track into its fixed byte range in a shared track-major signal file.
///
/// The output file has shape `(n_tracks, n_intervals, n_bins)` and each track
/// owns one contiguous range. This lets Rayon process tracks in parallel without
/// synchronizing writes.
pub(crate) fn process_one_track_at_offset(
    bw_path: &str,
    intervals: &[(String, u32, u32)],
    n_bins: usize,
    out_path: &str,
    offset: u64,
    len_bytes: u64,
    is_minus: bool,
) -> Result<(), String> {
    use std::io::{Seek, SeekFrom};

    let mut reader =
        BigWigRead::open_file(bw_path).map_err(|e| format!("Cannot open {bw_path}: {e}"))?;

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(out_path)
        .map_err(|e| format!("Cannot open {out_path}: {e}"))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("Seek error: {e}"))?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

    let (neg, nz) = write_track_rows(&mut reader, &mut writer, intervals, n_bins, is_minus)?;
    writer.flush().map_err(|e| format!("Flush error: {e}"))?;
    drop(writer);

    if is_minus && nz > 0 && neg as f32 / nz as f32 >= 0.8 {
        negate_file_range_inplace(out_path, offset, len_bytes)?;
    }

    Ok(())
}

/// Read one complete label row from a sample-major raw signal file.
///
/// This supports the compatibility writer. The production builder writes
/// track-major signals and uses `read_track_major_labels_batch`.
pub(crate) fn read_sample_labels(
    signal_file: &std::fs::File,
    local_idx: usize,
    n_tracks: usize,
    n_bins: usize,
) -> Result<Vec<f32>, String> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;

    let n_values = n_tracks
        .checked_mul(n_bins)
        .ok_or_else(|| "label shape is too large".to_string())?;
    let n_bytes = n_values
        .checked_mul(4)
        .ok_or_else(|| "label byte count is too large".to_string())?;
    let offset = (local_idx as u64)
        .checked_mul(n_bytes as u64)
        .ok_or_else(|| "label offset is too large".to_string())?;
    let mut bytes = vec![0u8; n_bytes];

    #[cfg(unix)]
    signal_file
        .read_exact_at(&mut bytes, offset)
        .map_err(|e| format!("Cannot read signal row {local_idx}: {e}"))?;

    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut fh = signal_file
            .try_clone()
            .map_err(|e| format!("Cannot clone signal handle: {e}"))?;
        fh.seek(SeekFrom::Start(offset))
            .map_err(|e| format!("Cannot seek signal row {local_idx}: {e}"))?;
        fh.read_exact(&mut bytes)
            .map_err(|e| format!("Cannot read signal row {local_idx}: {e}"))?;
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Read a batch of sample labels from the production track-major signal file.
///
/// Input layout is `(n_tracks, n_all_samples, n_bins)`. Output layout is
/// `(batch_rows, n_tracks, n_bins)` so it can be wrapped directly as the Arrow
/// `labels` Array2D column. If sample indices are contiguous, each track is read
/// with one larger positioned read rather than one read per sample.
pub(crate) fn read_track_major_labels_batch(
    signal_file: &std::fs::File,
    sample_indices: &[usize],
    n_all_samples: usize,
    n_tracks: usize,
    n_bins: usize,
) -> Result<Vec<f32>, String> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;

    let rows = sample_indices.len();
    let row_values = n_tracks
        .checked_mul(n_bins)
        .ok_or_else(|| "label row shape is too large".to_string())?;
    let mut labels = vec![0.0f32; rows * row_values];
    let bin_bytes = n_bins
        .checked_mul(4)
        .ok_or_else(|| "label byte count is too large".to_string())?;

    let contiguous = !sample_indices.is_empty()
        && sample_indices[sample_indices.len() - 1] - sample_indices[0] + 1 == sample_indices.len();

    for track_idx in 0..n_tracks {
        if contiguous {
            let first_sample = sample_indices[0];
            let offset_values = track_idx
                .checked_mul(n_all_samples)
                .and_then(|v| v.checked_add(first_sample))
                .and_then(|v| v.checked_mul(n_bins))
                .ok_or_else(|| "track-major offset is too large".to_string())?;
            let mut bytes = vec![0u8; rows * bin_bytes];

            #[cfg(unix)]
            signal_file
                .read_exact_at(&mut bytes, (offset_values * 4) as u64)
                .map_err(|e| format!("Cannot read track-major labels: {e}"))?;

            #[cfg(not(unix))]
            {
                use std::io::{Read, Seek, SeekFrom};
                let mut fh = signal_file
                    .try_clone()
                    .map_err(|e| format!("Cannot clone signal handle: {e}"))?;
                fh.seek(SeekFrom::Start((offset_values * 4) as u64))
                    .map_err(|e| format!("Cannot seek track-major labels: {e}"))?;
                fh.read_exact(&mut bytes)
                    .map_err(|e| format!("Cannot read track-major labels: {e}"))?;
            }

            for row_idx in 0..rows {
                let src_start = row_idx * bin_bytes;
                let dst_start = (row_idx * n_tracks + track_idx) * n_bins;
                for (bin_idx, b) in bytes[src_start..src_start + bin_bytes]
                    .chunks_exact(4)
                    .enumerate()
                {
                    labels[dst_start + bin_idx] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
        } else {
            let mut bytes = vec![0u8; bin_bytes];
            for (row_idx, sample_idx) in sample_indices.iter().copied().enumerate() {
                let offset_values = track_idx
                    .checked_mul(n_all_samples)
                    .and_then(|v| v.checked_add(sample_idx))
                    .and_then(|v| v.checked_mul(n_bins))
                    .ok_or_else(|| "track-major offset is too large".to_string())?;

                #[cfg(unix)]
                signal_file
                    .read_exact_at(&mut bytes, (offset_values * 4) as u64)
                    .map_err(|e| format!("Cannot read track-major labels: {e}"))?;

                #[cfg(not(unix))]
                {
                    use std::io::{Read, Seek, SeekFrom};
                    let mut fh = signal_file
                        .try_clone()
                        .map_err(|e| format!("Cannot clone signal handle: {e}"))?;
                    fh.seek(SeekFrom::Start((offset_values * 4) as u64))
                        .map_err(|e| format!("Cannot seek track-major labels: {e}"))?;
                    fh.read_exact(&mut bytes)
                        .map_err(|e| format!("Cannot read track-major labels: {e}"))?;
                }

                let dst_start = (row_idx * n_tracks + track_idx) * n_bins;
                for (bin_idx, b) in bytes.chunks_exact(4).enumerate() {
                    labels[dst_start + bin_idx] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
        }
    }

    Ok(labels)
}
