use arrow_ipc::{writer::IpcWriteOptions, CompressionType, MetadataVersion};
use std::time::{Duration, Instant};

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

pub(crate) fn ipc_write_options(compression: &str) -> Result<IpcWriteOptions, String> {
    let codec = match compression.to_ascii_lowercase().as_str() {
        "" | "none" | "uncompressed" => None,
        "zstd" => Some(CompressionType::ZSTD),
        "lz4" | "lz4_frame" => Some(CompressionType::LZ4_FRAME),
        other => {
            return Err(format!(
                "Unsupported Arrow compression '{other}'. Use 'zstd', 'lz4', or 'none'."
            ))
        }
    };

    IpcWriteOptions::try_new(8, false, MetadataVersion::V5)
        .and_then(|opts| opts.try_with_compression(codec))
        .map_err(|e| e.to_string())
}
