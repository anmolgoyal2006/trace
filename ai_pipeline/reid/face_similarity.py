"""
face_similarity.py — Face similarity utilities for TRACE Re-ID
==============================================================
Parallel to similarity.py (cosine_similarity) and track_aggregation.py
(aggregate_by_track), but operates on the face_embedding field produced
by face_embed.py.

Public API
----------
    face_cosine_similarity(a, b)     -> Optional[float]
    aggregate_face_by_track(records) -> dict[int | str, dict]

Design decisions
----------------
- face_cosine_similarity() returns None when either embedding is null
  (face not detected), instead of 0.0.  The caller must handle None
  explicitly — treating a missing face as "zero similarity" would silently
  bias the fused score downward for occluded or rear-facing persons.

- aggregate_face_by_track() adds face_coverage (fraction of crops with a
  detected face) alongside the same max / mean_top3 statistics used by
  aggregate_by_track().  A track's face signal should only be trusted when
  face_coverage >= some threshold (0.3 in matching_service.py).

- All return values are plain Python float / int — directly JSON-serialisable.

- Pure NumPy / pure Python — no GPU, no torchreid, no ONNX runtime.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional, Sequence, Union

import numpy as np

# Type alias compatible with similarity.py
_VectorLike = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# face_cosine_similarity
# ---------------------------------------------------------------------------

def face_cosine_similarity(
    a: Optional[_VectorLike],
    b: Optional[_VectorLike],
) -> Optional[float]:
    """
    Compute cosine similarity between two face embeddings.

    Mirrors similarity.cosine_similarity() but accepts null embeddings:
    returns None when either *a* or *b* is None (face not detected in
    that crop / track).  This lets the caller skip face fusion rather
    than silently penalising occluded or rear-facing persons.

    Args:
        a: First face embedding — 512-dim list/array, or None.
        b: Second face embedding — 512-dim list/array, or None.

    Returns:
        Cosine similarity as a Python float in [-1.0, 1.0], or None if
        either embedding is None.
        Returns 0.0 when both inputs are present but one is the zero vector.

    Raises:
        ValueError: if both embeddings are non-null but have different lengths
                    or zero length.

    Examples:
        >>> face_cosine_similarity(None, [0.1, 0.2])
        None
        >>> face_cosine_similarity([1.0, 0.0], [0.0, 1.0])
        0.0
        >>> face_cosine_similarity([1.0, 0.0], [1.0, 0.0])
        1.0
    """
    # Null guard — either embedding missing → cannot compute similarity
    if a is None or b is None:
        return None

    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()

    if a_arr.size == 0 or b_arr.size == 0:
        raise ValueError("Face embedding vectors must be non-empty.")
    if a_arr.size != b_arr.size:
        raise ValueError(
            f"Face embedding length mismatch: {a_arr.size} vs {b_arr.size}."
        )

    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))

    # Zero-vector guard — undefined cosine, return safe fallback
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    raw = float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
    # Clamp to absorb floating-point rounding beyond [-1, 1]
    return float(max(-1.0, min(1.0, raw)))


# ---------------------------------------------------------------------------
# aggregate_face_by_track
# ---------------------------------------------------------------------------

def aggregate_face_by_track(
    face_records: list[dict[str, Any]],
) -> dict[int | str, dict[str, float | int | None]]:
    """
    Group face embedding records by track and compute per-track statistics.

    Mirrors aggregate_by_track() from track_aggregation.py but operates on
    the ``face_embedding`` and ``face_detected`` fields from face_embed.py
    output.  Adds ``face_coverage`` — the fraction of crops in this track
    where a face was successfully detected.

    Parameters
    ----------
    face_records:
        A list of dicts, each containing at minimum:
            - ``track_id``       (int or str)
            - ``face_detected``  (bool)
            - ``face_embedding`` (list[float] | None)
        Additional fields are silently ignored.

    Returns
    -------
    dict
        Keys are track IDs (preserving the original type — int or str).
        Values are dicts with exactly these keys:

        ``max_face_sim``         float | None
            Highest single-crop face-to-face cosine similarity for the track,
            computed against the mean embedding of all detected-face crops.
            None when no face was detected for this track.

        ``mean_top3_face_sim``   float | None
            Mean of the top-3 per-crop similarities.
            None when no face was detected for this track.

        ``face_coverage``        float
            Fraction of crops in this track where face_detected == True.
            Always in [0.0, 1.0].

        ``num_observations``     int
            Total number of crops contributing to this track's record.

        ``num_face_detected``    int
            Number of crops where face_detected == True.

    Raises
    ------
    TypeError
        If ``face_records`` is not a list.
    ValueError
        If any record is missing ``track_id`` or ``face_detected``.

    Notes
    -----
    - Empty input returns an empty dict without raising.
    - Similarity values are computed between each detected-face crop and the
      mean face embedding for that track.  This gives a stable "intra-track
      consistency" measure analogous to what aggregate_by_track() produces
      with the body embeddings.
    - All returned numeric values are plain Python float / int —
      never NumPy scalar types — so the dict is directly JSON-serialisable.
    """
    if not isinstance(face_records, list):
        raise TypeError(
            f"face_records must be a list, got {type(face_records).__name__}"
        )

    # ---- validate and group by track_id ----------------------------------
    #   detected_embeddings: track_id → list of embedding arrays (non-null)
    #   total_counts:        track_id → total crop count (detected + missed)

    detected_embeddings: dict[Any, list[np.ndarray]] = defaultdict(list)
    total_counts:        dict[Any, int]               = defaultdict(int)

    for i, rec in enumerate(face_records):
        if "track_id" not in rec:
            raise ValueError(f"Record at index {i} is missing 'track_id'")
        if "face_detected" not in rec:
            raise ValueError(f"Record at index {i} is missing 'face_detected'")

        track_id = rec["track_id"]
        total_counts[track_id] += 1

        if rec["face_detected"]:
            emb = rec.get("face_embedding")
            if emb is not None:
                detected_embeddings[track_id].append(
                    np.asarray(emb, dtype=np.float64)
                )

    # ---- empty input fast-path -------------------------------------------
    if not total_counts:
        return {}

    # ---- compute per-track statistics ------------------------------------
    result: dict[Any, dict[str, float | int | None]] = {}

    for track_id, n_total in total_counts.items():
        embs = detected_embeddings.get(track_id, [])
        n_detected = len(embs)

        face_coverage = float(n_detected) / float(n_total)

        if n_detected == 0:
            # No face detected for this track — all similarity fields are null
            result[track_id] = {
                "max_face_sim":      None,
                "mean_top3_face_sim": None,
                "face_coverage":     float(face_coverage),
                "num_observations":  int(n_total),
                "num_face_detected": int(n_detected),
            }
            continue

        if n_detected == 1:
            # Only one face — self-similarity is 1.0; return that trivially
            result[track_id] = {
                "max_face_sim":       1.0,
                "mean_top3_face_sim": 1.0,
                "face_coverage":      float(face_coverage),
                "num_observations":   int(n_total),
                "num_face_detected":  int(n_detected),
            }
            continue

        # Compute mean (centroid) embedding for the track
        stacked  = np.stack(embs, axis=0)          # (N, 512)
        centroid = stacked.mean(axis=0)             # (512,)
        norm_c   = float(np.linalg.norm(centroid))
        if norm_c > 0.0:
            centroid /= norm_c

        # Per-crop similarity to centroid
        per_crop_sims: list[float] = []
        for emb in embs:
            sim = face_cosine_similarity(centroid, emb)
            if sim is not None:
                per_crop_sims.append(sim)

        if not per_crop_sims:
            result[track_id] = {
                "max_face_sim":       None,
                "mean_top3_face_sim": None,
                "face_coverage":      float(face_coverage),
                "num_observations":   int(n_total),
                "num_face_detected":  int(n_detected),
            }
            continue

        max_sim   = max(per_crop_sims)
        sorted_desc = sorted(per_crop_sims, reverse=True)
        top_k       = sorted_desc[:3]
        mean_top3   = sum(top_k) / len(top_k)

        result[track_id] = {
            "max_face_sim":       float(max_sim),
            "mean_top3_face_sim": float(mean_top3),
            "face_coverage":      float(face_coverage),
            "num_observations":   int(n_total),
            "num_face_detected":  int(n_detected),
        }

    return result
