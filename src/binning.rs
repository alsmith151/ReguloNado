use bigtools::{BBIFileRead, BigWigRead};

/// Statistics from a binning operation.
///
/// Tracks the number of direct binning calls and the total number of BigWig intervals
/// processed across all those calls. Intended for profiling to understand BigWig
/// iteration patterns.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BinningUsage {
    pub direct_calls: u64,
    pub direct_intervals: u64,
}

impl BinningUsage {
    fn direct(intervals: usize) -> Self {
        Self {
            direct_calls: 1,
            direct_intervals: intervals as u64,
        }
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.direct_calls += other.direct_calls;
        self.direct_intervals += other.direct_intervals;
    }
}

/// Reusable scratch buffers for binning operations.
///
/// Holds vectors for accumulating bin sums (f64) and covered bases per bin (u64),
/// allocated lazily and resized as needed. Passed through per-chromosome and per-thread
/// operations to avoid repeated allocations in tight loops.
#[derive(Default)]
pub(crate) struct BinningScratch {
    sums: Vec<f64>,
    covered: Vec<u64>,
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
}


/// Bin a BigWig region into `n_bins` equal-width bins (mean coverage per bin).
///
/// Writes into the caller-supplied `out` slice (length == n_bins).
/// Scratch buffers are allocated lazily inside `scratch` so only the
/// required data structures are paid for.
///
/// Uses a direct interval-to-bin overlap accumulator that keeps
/// exact mean-over-covered-bases semantics without materializing a per-base slab.
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

