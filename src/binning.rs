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

/// How a bin's mean is formed and what a bin with no usable bases holds.
///
/// Coverage BigWigs often omit zero-coverage stretches entirely, so a base with no
/// record is zero coverage, not missing data. The default therefore divides by the
/// bin's in-contig width (absent records contribute 0) and reserves "missing" for bins
/// that lie wholly past the contig end or whose every in-contig base is NaN-valued.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BinningOptions {
    /// Divide by the bases that have a (non-NaN) record instead of the in-contig bin
    /// width. The pre-2026-09 behaviour: inflates sparse bins in BigWigs without
    /// explicit zero records.
    pub mean_over_covered_bases: bool,
    /// Write NaN (rather than 0.0) for missing bins, so training can mask them.
    pub missing_as_nan: bool,
}

impl Default for BinningOptions {
    fn default() -> Self {
        Self {
            mean_over_covered_bases: false,
            missing_as_nan: true,
        }
    }
}

impl BinningOptions {
    pub(crate) fn missing_value(&self) -> f32 {
        if self.missing_as_nan {
            f32::NAN
        } else {
            0.0
        }
    }
}

/// Reusable scratch buffers for binning operations.
///
/// Holds vectors for accumulating bin sums (f64), covered bases per bin (u64) and
/// NaN-valued bases per bin (u64), allocated lazily and resized as needed. Passed
/// through per-chromosome and per-thread operations to avoid repeated allocations in
/// tight loops.
#[derive(Default)]
pub(crate) struct BinningScratch {
    sums: Vec<f64>,
    covered: Vec<u64>,
    nan_bases: Vec<u64>,
}

impl BinningScratch {
    fn direct_buffers(&mut self, n_bins: usize) -> (&mut [f64], &mut [u64], &mut [u64]) {
        if self.sums.len() < n_bins {
            self.sums.resize(n_bins, 0.0);
        }
        if self.covered.len() < n_bins {
            self.covered.resize(n_bins, 0);
        }
        if self.nan_bases.len() < n_bins {
            self.nan_bases.resize(n_bins, 0);
        }

        let sums = &mut self.sums[..n_bins];
        sums.fill(0.0);
        let covered = &mut self.covered[..n_bins];
        covered.fill(0);
        let nan_bases = &mut self.nan_bases[..n_bins];
        nan_bases.fill(0);
        (sums, covered, nan_bases)
    }
}

/// Add `overlap` bases of `value` to one bin, routing NaN values to the NaN count.
#[inline]
fn accumulate(bin: usize, value: f64, overlap: u64, sums: &mut [f64], covered: &mut [u64], nan_bases: &mut [u64]) {
    if value.is_nan() {
        nan_bases[bin] += overlap;
    } else {
        sums[bin] += value * overlap as f64;
        covered[bin] += overlap;
    }
}

/// Bin a BigWig region into `n_bins` equal-width bins (mean signal per bin).
///
/// Writes into the caller-supplied `out` slice (length == n_bins).
/// Scratch buffers are allocated lazily inside `scratch` so only the
/// required data structures are paid for.
///
/// Uses a direct interval-to-bin overlap accumulator without materializing a per-base
/// slab. The denominator and the value of missing bins follow `options` (see
/// [`BinningOptions`]); bins past the contig end count as missing.
pub(crate) fn bin_region_into<R: BBIFileRead>(
    reader: &mut BigWigRead<R>,
    chrom: &str,
    region_start: u32,
    region_end: u32,
    out: &mut [f32],
    scratch: &mut BinningScratch,
    options: BinningOptions,
) -> Result<BinningUsage, String> {
    let n_bins = out.len();
    if n_bins == 0 || region_end <= region_start {
        out.fill(options.missing_value());
        return Ok(BinningUsage::default());
    }
    let region_width = region_end - region_start;
    // A chromosome absent from this BigWig's header is left to get_interval to report.
    let chrom_len = reader
        .info()
        .chrom_info
        .iter()
        .find(|info| info.name == chrom)
        .map_or(u32::MAX, |info| info.length);

    let iter = reader
        .get_interval(chrom, region_start, region_end)
        .map_err(|error| {
            format!("failed to read BigWig interval {chrom}:{region_start}-{region_end}: {error}")
        })?;

    let mut interval_count = 0usize;
    let (sums, covered, nan_bases) = scratch.direct_buffers(n_bins);

    if region_width as usize % n_bins == 0 {
        let bin_bp = (region_width as usize) / n_bins;
        for val_result in iter {
            let val = val_result.map_err(|error| {
                format!(
                    "failed while reading BigWig intervals for {chrom}:{region_start}-{region_end}: {error}"
                )
            })?;
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
                accumulate(first_bin, value, overlap, sums, covered, nan_bases);
                continue;
            }

            let first_bin_end = (first_bin + 1) * bin_bp;
            let first_overlap = (first_bin_end - rel_start) as u64;
            accumulate(first_bin, value, first_overlap, sums, covered, nan_bases);

            for bin in first_bin + 1..last_bin {
                accumulate(bin, value, bin_bp as u64, sums, covered, nan_bases);
            }

            let last_bin_start = last_bin * bin_bp;
            let last_overlap = (rel_end - last_bin_start) as u64;
            accumulate(last_bin, value, last_overlap, sums, covered, nan_bases);
        }
    } else {
        let bin_bp = region_width as f64 / n_bins as f64;
        for val_result in iter {
            let val = val_result.map_err(|error| {
                format!(
                    "failed while reading BigWig intervals for {chrom}:{region_start}-{region_end}: {error}"
                )
            })?;
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
                accumulate(first_bin, val.value as f64, ovl_len, sums, covered, nan_bases);
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
                        accumulate(bin, val.value as f64, ovl_len, sums, covered, nan_bases);
                    }
                }
            }
        }
    }

    let exact_bin_bp = region_width as usize % n_bins == 0;
    for i in 0..n_bins {
        let (bin_abs_start, bin_abs_end) = if exact_bin_bp {
            let bin_bp = region_width / n_bins as u32;
            (region_start + i as u32 * bin_bp, region_start + (i as u32 + 1) * bin_bp)
        } else {
            let bin_bp = region_width as f64 / n_bins as f64;
            let end = if i + 1 >= n_bins {
                region_end
            } else {
                region_start + ((i + 1) as f64 * bin_bp) as u32
            };
            (region_start + (i as f64 * bin_bp) as u32, end)
        };
        let in_contig = bin_abs_end.min(chrom_len).saturating_sub(bin_abs_start) as u64;
        if in_contig == 0 || nan_bases[i] >= in_contig {
            out[i] = options.missing_value();
            continue;
        }
        let denominator = if options.mean_over_covered_bases {
            covered[i]
        } else {
            in_contig - nan_bases[i]
        };
        out[i] = if denominator > 0 {
            (sums[i] / denominator as f64) as f32
        } else {
            0.0
        };
    }

    Ok(BinningUsage::direct(interval_count))
}
