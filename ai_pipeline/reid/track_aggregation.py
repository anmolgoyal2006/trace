"""
Phase 5.3.1 — Track-Level Aggregation Statistics
==================================================
Accepts crop-level similarity records (each record has at least a
``track_id`` and a ``similarity`` field) and returns per-track aggregated
statistics.

This module is pure CPU / pure Python — no GPU, no NumPy, no torchreid.
It is designed to be imported into a Google Colab notebook that has already
produced the crop-level similarity output from Phase 5.2.

Aggregation statistics per track
---------------------------------
max_similarity
    Maximum crop-level cosine similarity observed for that track.

mean_similarity
    Arithmetic mean of ALL crop-level similarities for that track.

mean_top3_similarity
    Mean of the top-3 highest similarities for that track.
    If the track has fewer than 3 crops, the mean is taken over all
    available crops (i.e. no padding or substitution is applied).

num_observations
    Number of gallery crops that contributed to that track's statistics.

Return types
------------
All values in the returned dict are plain Python ``int`` or ``float``.
No NumPy scalar types are used, so the result is directly JSON-serialisable.

Usage example
-------------
>>> from ai_pipeline.reid.track_aggregation import aggregate_by_track
>>> records = [
...     {"track_id": 29, "similarity": 0.79},
...     {"track_id": 29, "similarity": 0.70},
...     {"track_id": 29, "similarity": 0.85},
...     {"track_id": 66, "similarity": 0.81},
...     {"track_id": 66, "similarity": 0.75},
... ]
>>> result = aggregate_by_track(records)
>>> result[29]["max_similarity"]
0.85
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def aggregate_by_track(
    records: list[dict[str, Any]],
) -> dict[int | str, dict[str, float | int]]:
    """Group crop-level similarity records by track and compute statistics.

    Parameters
    ----------
    records:
        A list of dicts, each containing at minimum:
            - ``track_id``  (int or str) — identifier of the track
            - ``similarity`` (float)     — cosine similarity score for that crop
        Additional fields are silently ignored.

    Returns
    -------
    dict
        Keys are track IDs (preserving the original type — int or str).
        Values are dicts with exactly these keys:

        ``max_similarity``      float — highest single-crop similarity
        ``mean_similarity``     float — mean over all crops for the track
        ``mean_top3_similarity``float — mean of the top-3 crops (or all if < 3)
        ``num_observations``    int   — number of contributing crops

    Raises
    ------
    TypeError
        If ``records`` is not a list.
    ValueError
        If any record is missing ``track_id`` or ``similarity``.

    Notes
    -----
    - Empty input returns an empty dict; it does NOT raise.
    - All returned numeric values are plain Python ``float`` / ``int``,
      never NumPy scalar types, so the dict is directly JSON-serialisable.
    """
    if not isinstance(records, list):
        raise TypeError(
            f"records must be a list, got {type(records).__name__}"
        )

    # Validate and group ------------------------------------------------
    grouped: dict[Any, list[float]] = defaultdict(list)
    for i, rec in enumerate(records):
        if "track_id" not in rec:
            raise ValueError(f"Record at index {i} is missing 'track_id'")
        if "similarity" not in rec:
            raise ValueError(f"Record at index {i} is missing 'similarity'")
        track_id = rec["track_id"]
        similarity = float(rec["similarity"])   # normalise to plain float
        grouped[track_id].append(similarity)

    # Empty input: return empty dict ------------------------------------
    if not grouped:
        return {}

    # Compute statistics ------------------------------------------------
    result: dict[Any, dict[str, float | int]] = {}

    for track_id, sims in grouped.items():
        n = len(sims)

        # max
        max_sim = max(sims)

        # mean over all
        mean_sim = sum(sims) / n

        # mean of top-3 (or all if fewer than 3)
        sorted_desc = sorted(sims, reverse=True)
        top_k = sorted_desc[:3]          # slice gives <= 3 elements
        mean_top3 = sum(top_k) / len(top_k)

        result[track_id] = {
            "max_similarity":       float(max_sim),
            "mean_similarity":      float(mean_sim),
            "mean_top3_similarity": float(mean_top3),
            "num_observations":     int(n),
        }

    return result
