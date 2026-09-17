"""Count label space: raw-count targets with a per-track exposure offset.

Selected by ``data.label_space: counts``. Stored signal (mean coverage per bin) is
converted to count units at read time and never scaled, background-subtracted,
clipped or squashed. The model predicts an exposure-normalised rate ``r``; the
likelihood terms see ``mu = r * exposure`` against the raw counts (a GLM offset), while
cross-track comparisons (contrast loss, metrics, plots, prediction BigWigs) use the
normalised space ``counts / exposure``.

``count_unit`` — what one label unit is:
    - ``fragments``: ``mean_coverage * bin_size / fragment_length`` ≈ fragments per
      bin, the scale at which Poisson variance roughly holds for fragment pileups.
    - ``coverage_sum``: ``mean_coverage * bin_size``, summed per-base coverage
      (Borzoi's convention).
    - ``mean_coverage``: the stored value unchanged.

``exposure`` — count units per model-output unit:
    - ``anchor``: the track's anchor reference, so an output of 1.0 is the
      housekeeping-promoter level (background is not subtracted).
    - ``anchor_minus_background``: anchor reference minus background median.
    - ``library_size``: mapped reads in millions (outputs are per-million).
    - ``none``: 1.0, so the model predicts count units directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

LABEL_SPACES = ("transformed", "counts")
COUNT_UNITS = ("fragments", "coverage_sum", "mean_coverage")
EXPOSURES = ("anchor", "anchor_minus_background", "library_size", "none")


@dataclass(frozen=True)
class CountLabelSpace:
    """Per-track factors for the count label space, shape ``(n_tracks,)`` each."""

    # Stored mean coverage -> count units.
    count_factors: np.ndarray
    # Count units per model-output unit.
    exposure: np.ndarray


def validate_label_space(label_space: str, count_unit: str, exposure: str) -> None:
    for name, value, allowed in (
        ("data.label_space", label_space, LABEL_SPACES),
        ("data.count_unit", count_unit, COUNT_UNITS),
        ("data.exposure", exposure, EXPOSURES),
    ):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")


def _names(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> str:
    names = [str(records[i].get("track_name", i)) for i in indices]
    return ", ".join(names[:5]) + (f" (+{len(names) - 5} more)" if len(names) > 5 else "")


def _track_field(records: Sequence[Mapping[str, Any]], field: str, *, purpose: str) -> np.ndarray:
    values = np.array(
        [float(record[field]) if record.get(field) is not None else np.nan for record in records],
        dtype=np.float64,
    )
    missing = np.flatnonzero(~np.isfinite(values))
    if missing.size:
        raise ValueError(
            f"{purpose} needs '{field}' for every track; missing for: {_names(records, missing)}"
        )
    return values


def _fragment_lengths(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """``fragment_length`` per track, or ``genome_sum / library_size`` where it is absent.

    Summed genome-wide coverage over the number of fragments is the mean fragment
    length when the BigWig is an unscaled fragment pileup.
    """
    lengths = []
    for record in records:
        if record.get("fragment_length") is not None:
            lengths.append(float(record["fragment_length"]))
        elif record.get("genome_sum") and record.get("library_size"):
            lengths.append(float(record["genome_sum"]) / float(record["library_size"]))
        else:
            lengths.append(np.nan)
    lengths_array = np.asarray(lengths, dtype=np.float64)
    bad = np.flatnonzero(~(np.isfinite(lengths_array) & (lengths_array > 0)))
    if bad.size:
        raise ValueError(
            "data.count_unit=fragments needs a positive 'fragment_length' per track (add it "
            "via inputs.track_annotations), or both 'scale_library_size' and "
            f"'fp_genome_sum' to derive it; missing for: {_names(records, bad)}"
        )
    return lengths_array


def resolve_count_label_space(
    records: Sequence[Mapping[str, Any]],
    *,
    count_unit: str,
    exposure: str,
    bin_size: int,
) -> CountLabelSpace:
    """Compute count-unit conversion and exposure for every track in ``records``."""
    validate_label_space("counts", count_unit, exposure)
    n_tracks = len(records)

    if count_unit == "mean_coverage":
        count_factors = np.ones(n_tracks, dtype=np.float64)
    elif count_unit == "coverage_sum":
        count_factors = np.full(n_tracks, float(bin_size), dtype=np.float64)
    else:
        count_factors = float(bin_size) / _fragment_lengths(records)

    purpose = f"data.exposure={exposure}"
    if exposure == "none":
        exposure_counts = np.ones(n_tracks, dtype=np.float64)
    elif exposure == "library_size":
        library_millions = _track_field(records, "library_size", purpose=purpose) / 1e6
        exposure_counts = library_millions * count_factors
    else:
        reference = _track_field(records, "anchor_reference", purpose=purpose)
        if exposure == "anchor_minus_background":
            reference = reference - _track_field(records, "background", purpose=purpose)
        exposure_counts = reference * count_factors

    bad = np.flatnonzero(~(np.isfinite(exposure_counts) & (exposure_counts > 0)))
    if bad.size:
        raise ValueError(f"{purpose} is not positive for tracks: {_names(records, bad)}")
    return CountLabelSpace(
        count_factors=count_factors.astype(np.float32),
        exposure=exposure_counts.astype(np.float32),
    )
