use std::sync::OnceLock;

use bigtools::{BBIFileRead, BigWigRead};
use wide::CmpEq;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BinningUsage {
    pub direct_calls: u64,
    pub dense_calls: u64,
    pub direct_intervals: u64,
    pub dense_intervals: u64,
}

impl BinningUsage {
    fn direct(intervals: usize) -> Self {
        Self {
            direct_calls: 1,
            dense_calls: 0,
            direct_intervals: intervals as u64,
            dense_intervals: 0,
        }
    }

    fn dense(intervals: usize) -> Self {
        Self {
            direct_calls: 0,
            dense_calls: 1,
            direct_intervals: 0,
            dense_intervals: intervals as u64,
        }
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.direct_calls += other.direct_calls;
        self.dense_calls += other.dense_calls;
        self.direct_intervals += other.direct_intervals;
        self.dense_intervals += other.dense_intervals;
    }
}

#[derive(Default)]
pub(crate) struct BinningScratch {
    sums: Vec<f64>,
    covered: Vec<u64>,
    values_buf: Vec<f32>,
}

impl BinningScratch {
    fn direct_buffers(&mut self, n_bins: usize) -> (&mut [f64], &mut [u64]) {
        if self.sums.len() < n_bins {
            self.sums.resize(n_bins, 0.0);
        }
        if self.covered.len() < n_bins {
            self.covered.resize(n_bins, 0);
        }

        let sums = &mut self.sums[..n_bins];
        sums.fill(0.0);
        let covered = &mut self.covered[..n_bins];
        covered.fill(0);
        (sums, covered)
    }

    fn dense_buffer(&mut self, region_width: usize) -> &mut [f32] {
        self.values_buf.resize(region_width, f32::NAN);
        self.values_buf.fill(f32::NAN);
        &mut self.values_buf[..region_width]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BinningMode {
    Direct,
    Dense,
}

fn dense_binner_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("REGULONADO_BINNING_FORCE_DENSE")
            .ok()
            .map(|value| {
                let trimmed = value.trim();
                !trimmed.is_empty()
                    && trimmed != "0"
                    && !trimmed.eq_ignore_ascii_case("false")
            })
            .unwrap_or(false)
    })
}

fn select_binning_mode(region_width: usize, n_bins: usize) -> BinningMode {
    if region_width % n_bins == 0 && dense_binner_enabled() {
        BinningMode::Dense
    } else {
        BinningMode::Direct
    }
}

/// Bin a BigWig region into `n_bins` equal-width bins (mean coverage per bin).
///
/// Writes into the caller-supplied `out` slice (length == n_bins).
/// Scratch buffers are allocated lazily inside `scratch` so the selected path
/// only pays for the data structures it actually uses.
///
/// The default path is a direct interval-to-bin overlap accumulator that keeps
/// exact mean-over-covered-bases semantics without materializing a per-base slab.
/// The older dense slab implementation remains available behind the
/// `REGULONADO_BINNING_FORCE_DENSE=1` diagnostic override.
pub(crate) fn bin_region_into<R: BBIFileRead>(
    reader: &mut BigWigRead<R>,
    chrom: &str,
    region_start: u32,
    region_end: u32,
    out: &mut [f32],
    scratch: &mut BinningScratch,
) -> BinningUsage {
    let n_bins = out.len();
    if n_bins == 0 || region_end <= region_start {
        out.fill(0.0);
        return BinningUsage::default();
    }
    let region_width = region_end - region_start;

    let iter = match reader.get_interval(chrom, region_start, region_end) {
        Ok(it) => it,
        Err(_) => {
            out.fill(0.0);
            return BinningUsage::default();
        }
    };

    match select_binning_mode(region_width as usize, n_bins) {
        BinningMode::Direct => {
            let mut interval_count = 0usize;
            let (sums, covered) = scratch.direct_buffers(n_bins);

            if region_width as usize % n_bins == 0 {
                let bin_bp = (region_width as usize) / n_bins;
                for val_result in iter {
                    let val = match val_result {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if val.value.is_nan() {
                        continue;
                    }
                    let v_start = val.start.max(region_start);
                    let v_end = val.end.min(region_end);
                    if v_start >= v_end {
                        continue;
                    }

                    interval_count += 1;

                    let rel_start = (v_start - region_start) as usize;
                    let rel_end = (v_end - region_start) as usize;
                    let first_bin = rel_start / bin_bp;
                    let last_bin = (rel_end - 1) / bin_bp;
                    let value = val.value as f64;

                    if first_bin == last_bin {
                        let overlap = (rel_end - rel_start) as u64;
                        sums[first_bin] += value * overlap as f64;
                        covered[first_bin] += overlap;
                        continue;
                    }

                    let first_bin_end = (first_bin + 1) * bin_bp;
                    let first_overlap = (first_bin_end - rel_start) as u64;
                    sums[first_bin] += value * first_overlap as f64;
                    covered[first_bin] += first_overlap;

                    if last_bin > first_bin + 1 {
                        let full_bin_coverage = bin_bp as u64;
                        let full_bin_sum = value * full_bin_coverage as f64;
                        for bin in first_bin + 1..last_bin {
                            sums[bin] += full_bin_sum;
                            covered[bin] += full_bin_coverage;
                        }
                    }

                    let last_bin_start = last_bin * bin_bp;
                    let last_overlap = (rel_end - last_bin_start) as u64;
                    sums[last_bin] += value * last_overlap as f64;
                    covered[last_bin] += last_overlap;
                }
            } else {
                let bin_bp = region_width as f64 / n_bins as f64;
                for val_result in iter {
                    let val = match val_result {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if val.value.is_nan() {
                        continue;
                    }
                    let v_start = val.start.max(region_start);
                    let v_end = val.end.min(region_end);
                    if v_start >= v_end {
                        continue;
                    }

                    interval_count += 1;

                    let rel_start = (v_start - region_start) as f64;
                    let rel_end = (v_end - region_start) as f64;
                    let first_bin = (rel_start / bin_bp) as usize;
                    let last_bin = ((rel_end / bin_bp).ceil() as usize).min(n_bins);
                    if first_bin + 1 == last_bin {
                        let ovl_len = (v_end - v_start) as u64;
                        sums[first_bin] += val.value as f64 * ovl_len as f64;
                        covered[first_bin] += ovl_len;
                    } else {
                        for bin in first_bin..last_bin {
                            let bin_abs_start = region_start + (bin as f64 * bin_bp) as u32;
                            let bin_abs_end = if bin + 1 >= n_bins {
                                region_end
                            } else {
                                region_start + ((bin + 1) as f64 * bin_bp) as u32
                            };
                            let ovl_start = v_start.max(bin_abs_start);
                            let ovl_end = v_end.min(bin_abs_end);
                            if ovl_start < ovl_end {
                                let ovl_len = (ovl_end - ovl_start) as u64;
                                sums[bin] += val.value as f64 * ovl_len as f64;
                                covered[bin] += ovl_len;
                            }
                        }
                    }
                }
            }

            for i in 0..n_bins {
                out[i] = if covered[i] > 0 {
                    (sums[i] / covered[i] as f64) as f32
                } else {
                    0.0
                };
            }

            BinningUsage::direct(interval_count)
        }
        BinningMode::Dense => {
            let mut interval_count = 0usize;
            let bin_bp = (region_width as usize) / n_bins;
            let values_buf = scratch.dense_buffer(region_width as usize);

            for val_result in iter {
                let val = match val_result {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if val.value.is_nan() {
                    continue;
                }
                let v_start = (val.start.max(region_start) - region_start) as usize;
                let v_end = (val.end.min(region_end) - region_start) as usize;
                if v_start < v_end {
                    interval_count += 1;
                    values_buf[v_start..v_end].fill(val.value);
                }
            }

            for bin in 0..n_bins {
                let slice = &values_buf[bin * bin_bp..(bin + 1) * bin_bp];
                let mut sum = 0.0f32;
                let mut cnt = 0.0f32;

                let mut chunks = slice.chunks_exact(8);
                for chunk in chunks.by_ref() {
                    let v = wide::f32x8::new(chunk.try_into().unwrap());
                    let valid = v.cmp_eq(v);
                    let clean = v & valid;
                    let ones = wide::f32x8::splat(1.0_f32) & valid;
                    sum += clean.reduce_add();
                    cnt += ones.reduce_add();
                }
                for &x in chunks.remainder() {
                    if !x.is_nan() {
                        sum += x;
                        cnt += 1.0;
                    }
                }

                out[bin] = if cnt > 0.0 { sum / cnt } else { 0.0 };
            }

            BinningUsage::dense(interval_count)
        }
    }
}

/// Convenience wrapper returning a Vec.
///
/// This exists for the NumPy-returning debug helper. Production extraction
/// reuses caller-owned row and scratch buffers through `bin_region_into`.
pub(crate) fn bin_region<R: BBIFileRead>(
    reader: &mut BigWigRead<R>,
    chrom: &str,
    region_start: u32,
    region_end: u32,
    n_bins: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_bins];
    let mut scratch = BinningScratch::default();
    let _ = bin_region_into(reader, chrom, region_start, region_end, &mut out, &mut scratch);
    out
}
