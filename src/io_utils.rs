use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use std::time::{Duration, Instant};

/// Log progress at most once per 30 seconds, or when finished.
///
/// Computes progress percentage, elapsed time, rate (items/s), and estimated time
/// to completion, printing to stderr in the format `[regulonado_rs] {label}: {done}/{total} ...`.
/// Updates the `last_log` timestamp if a log is printed. Intended for long-running tasks
/// to avoid spamming stderr while still giving visibility into progress.
pub(crate) fn maybe_log_progress(
    last_log: &mut Instant,
    started: Instant,
    label: &str,
    done: usize,
    total: usize,
) {
    let now = Instant::now();
    if done >= total || now.duration_since(*last_log) >= Duration::from_secs(30) {
        let elapsed = now.duration_since(started).as_secs_f64();
        let rate = if elapsed > 0.0 {
            done as f64 / elapsed
        } else {
            0.0
        };
        let eta = if rate > 0.0 && done < total {
            (total - done) as f64 / rate
        } else {
            0.0
        };
        eprintln!(
            "[regulonado_rs] {label}: {done}/{total} ({:.1}%) elapsed {:.1}s rate {:.2}/s eta {:.1}s",
            done as f64 * 100.0 / total.max(1) as f64,
            elapsed,
            rate,
            eta,
        );
        *last_log = now;
    }
}

/// Configure the global Rayon pool, warning if it was already initialised.
///
/// Rayon's global pool can only be built once per process. A second call — which happens
/// whenever a writer entry point is invoked more than once from the same interpreter —
/// fails, and the requested `n_threads` is quietly ignored. Previously that failure was
/// discarded with `.ok()`, so the thread count a caller passed had no effect and no
/// diagnostic. The pool still cannot be rebuilt, but at least the mismatch is now visible.
pub(crate) fn configure_global_rayon(n_threads: Option<usize>) {
    let Some(nt) = n_threads else {
        return;
    };
    if rayon::ThreadPoolBuilder::new()
        .num_threads(nt)
        .build_global()
        .is_err()
    {
        eprintln!(
            "[regulonado_rs] warning: requested n_threads={nt} but the global Rayon pool is \
             already initialised (it can only be configured once per process); continuing \
             with the existing pool of {} threads.",
            rayon::current_num_threads(),
        );
    }
}

/// Build Parquet writer properties: one row group per `rows_per_row_group` rows,
/// compressed with the given explicit ZSTD level. Page statistics are left at the
/// Parquet default (on), which also produces a page index.
pub(crate) fn parquet_writer_properties(
    rows_per_row_group: usize,
    zstd_level: i32,
) -> Result<WriterProperties, String> {
    let level = ZstdLevel::try_new(zstd_level)
        .map_err(|e| format!("Invalid zstd level {zstd_level}: {e}"))?;
    Ok(WriterProperties::builder()
        .set_max_row_group_size(rows_per_row_group.max(1))
        .set_compression(Compression::ZSTD(level))
        .build())
}
