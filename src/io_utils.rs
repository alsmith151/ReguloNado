use arrow_ipc::{CompressionType, MetadataVersion, writer::IpcWriteOptions};
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
