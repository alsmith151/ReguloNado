from array import array
from pathlib import Path

import numpy as np
from regulonado.counts.bam import (
    _DUPLICATE_FLAG,
    _EXCLUDED_FLAGS,
    _TN5_FORWARD_SHIFT,
    _TN5_REVERSE_SHIFT,
    CountSpec,
)


def reference_count_events(
    bam_path: str | Path, chrom: str, spec: CountSpec, *, threads: int = 1
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Stream one chromosome of a BAM into sorted event starts and ends.

    Returns:
        ``(starts, ends, n_paired, n_single)``: the sorted int64 start and
        (separately sorted) end coordinates of every event, and how many of
        them came from paired vs single-end reads.
    """
    import pysam

    # array("q") rather than lists: ~8 bytes per event instead of ~36, which
    # matters at tens of millions of events on the large chromosomes.
    starts = array("q")
    ends = array("q")
    n_paired = 0
    n_single = 0
    excluded = _EXCLUDED_FLAGS | (_DUPLICATE_FLAG if spec.drop_duplicates else 0)

    with pysam.AlignmentFile(str(bam_path), "rb", threads=threads) as bam:
        if chrom not in bam.references:
            return np.empty(0, np.int64), np.empty(0, np.int64), 0, 0
        for read in bam.fetch(chrom):
            if read.flag & excluded or read.mapping_quality < spec.min_mapq:
                continue
            paired = read.is_paired
            if paired:
                if spec.proper_pairs_only and not read.is_proper_pair:
                    continue
                if (
                    spec.max_fragment_length is not None
                    and abs(read.template_length) > spec.max_fragment_length
                ):
                    continue

            if spec.unit == "insertions":
                if read.is_reverse:
                    position = read.reference_end - 1
                    position += _TN5_REVERSE_SHIFT if spec.tn5_shift else 0
                else:
                    position = read.reference_start
                    position += _TN5_FORWARD_SHIFT if spec.tn5_shift else 0
                starts.append(position)
                ends.append(position + 1)
            elif paired:
                # One event per fragment: only the leftmost mate (TLEN > 0).
                if read.template_length <= 0:
                    continue
                starts.append(read.reference_start)
                ends.append(read.reference_start + read.template_length)
            else:
                start, end = read.reference_start, read.reference_end
                if spec.single_end_extension is not None:
                    if read.is_reverse:
                        start = end - spec.single_end_extension
                    else:
                        end = start + spec.single_end_extension
                starts.append(start)
                ends.append(end)

            if paired:
                n_paired += 1
            else:
                n_single += 1

    return (
        np.sort(np.frombuffer(starts, dtype=np.int64)),
        np.sort(np.frombuffer(ends, dtype=np.int64)),
        n_paired,
        n_single,
    )
