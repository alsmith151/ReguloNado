"""Run separate-process Rust/pysam benchmarks: python tests/bench_bam_counting.py BAM CHROM."""

import argparse
import json
import multiprocessing as mp
import resource
import sys
import time


def run(path, chrom, backend, connection):
    from bam_reference import reference_count_events
    from regulonado.counts.bam import CountSpec, count_events

    count = count_events if backend == "rust" else reference_count_events
    started = time.perf_counter()
    starts, ends, paired, single = count(path, chrom, CountSpec("fragments"))
    elapsed = time.perf_counter() - started
    import hashlib

    digest = hashlib.sha256(starts.tobytes() + ends.tobytes()).hexdigest()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    connection.send(
        {
            "backend": backend,
            "seconds": elapsed,
            "peak_rss_mb": peak / (1024**2 if sys.platform == "darwin" else 1024),
            "events": paired + single,
            "sha256": digest,
        }
    )
    connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bam")
    parser.add_argument("chrom")
    args = parser.parse_args()
    results = []
    for backend in ("pysam", "rust"):
        receiver, sender = mp.Pipe(duplex=False)
        process = mp.get_context("spawn").Process(
            target=run, args=(args.bam, args.chrom, backend, sender)
        )
        process.start()
        sender.close()
        results.append(receiver.recv())
        process.join()
        if process.exitcode:
            raise SystemExit(process.exitcode)
    assert results[0]["sha256"] == results[1]["sha256"]
    print(
        json.dumps(
            {"results": results, "speedup": results[0]["seconds"] / results[1]["seconds"]}, indent=2
        )
    )
