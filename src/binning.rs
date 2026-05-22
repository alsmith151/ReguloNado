use bigtools::{BBIFileRead, BigWigRead};
use wide::CmpEq;

/// Bin a BigWig region into `n_bins` equal-width bins (mean coverage per bin).
///
/// Writes into the caller-supplied `out` slice (length == n_bins).
/// `sums` and `covered` are scratch buffers of the same length, passed in so
/// the hot path can allocate them once per track rather than once per interval.
///
/// When the region width divides evenly by `n_bins` (the normal build case),
/// all bin-boundary math stays integer-only. The floating-point fallback is
/// retained for non-divisible diagnostic calls.
pub(crate) fn bin_region_into<R: BBIFileRead>(
    reader: &mut BigWigRead<R>,
    chrom: &str,
    region_start: u32,
    region_end: u32,
    out: &mut [f32],
    sums: &mut [f64],
    covered: &mut [u64],
    values_buf: &mut Vec<f32>,
) {
    let n_bins = out.len();
    let region_width = region_end - region_start;

    let iter = match reader.get_interval(chrom, region_start, region_end) {
        Ok(it) => it,
        Err(_) => {
            out.fill(0.0);
            return;
        }
    };

    if region_width as usize % n_bins == 0 {
        // Dense slab fill: one contiguous .fill() per BigWig interval, then
        // SIMD reduction over the in-RAM slab. Removes per-interval bin-overlap
        // math and lets the compiler vectorise the inner reduction loop.
        let bin_bp = (region_width as usize) / n_bins;

        values_buf.resize(region_width as usize, f32::NAN);
        values_buf.fill(f32::NAN);

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
                values_buf[v_start..v_end].fill(val.value);
            }
        }

        // SIMD mean reduction per bin: NaN lanes (uncovered) are zeroed and
        // excluded from the count via a branchless bitmask.
        for bin in 0..n_bins {
            let slice = &values_buf[bin * bin_bp..(bin + 1) * bin_bp];
            let mut sum = 0.0f32;
            let mut cnt = 0.0f32;

            let mut chunks = slice.chunks_exact(8);
            for chunk in chunks.by_ref() {
                let v = wide::f32x8::new(chunk.try_into().unwrap());
                let valid = v.cmp_eq(v); // all-1 bits where not NaN (NaN != NaN)
                let clean = v & valid; // zero NaN lanes
                let ones = wide::f32x8::splat(1.0_f32) & valid; // 1.0 or 0.0
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
    } else {
        // Non-divisible fallback: original interval-scan path retained for
        // diagnostic builds where region_width % n_bins != 0.
        sums.fill(0.0);
        covered.fill(0);
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
        for i in 0..n_bins {
            out[i] = if covered[i] > 0 {
                (sums[i] / covered[i] as f64) as f32
            } else {
                0.0
            };
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
    let mut sums = vec![0.0f64; n_bins];
    let mut covered = vec![0u64; n_bins];
    let mut values_buf: Vec<f32> = Vec::new();
    bin_region_into(
        reader,
        chrom,
        region_start,
        region_end,
        &mut out,
        &mut sums,
        &mut covered,
        &mut values_buf,
    );
    out
}
