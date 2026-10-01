"""
track_aggregation.py — Phase 5.3.1 — Track-Level Aggregation Statistics
=========================================================================

Accepts crop-level similarity records (each record has at least a
``track_id`` and a ``similarity`` field) and returns per-track aggregated
statistics.

Face-first policy (TRACE update)
----------------------------------
When a track has sufficient reliable face evidence, face similarity
is the primary identity score — body similarity is NOT blended.

Sufficient face evidence means:
  - face_coverage >= 0.3 (fraction of crops with face_detected=True)
  - At least one valid face similarity value in the track

If face evidence is sufficient:
    identity_mode = "face"
    identity_score = face_similarity ONLY

Otherwise:
    identity_mode = "body"
    identity_score = body/part Re-ID score (existing fallback)

This module is pure CPU / pure Python — no GPU, no NumPy, no torchreid.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


# Face-first policy threshold — mirrors matching_service._FACE_COVERAGE_MIN
_FACE_COVERAGE_MIN: float = 0.3


def aggregate_by_track(
    records: list[dict[str, Any]],
) -> dict[int | str, dict[str, float | int | bool]]:
    """Group crop-level similarity records by track and compute statistics.

    Parameters
    ----------
    records:
        A list of dicts, each containing at minimum:
            - ``track_id``       (int or str) — identifier of the track
            - ``similarity``     (float)     — cosine similarity score for that crop
            - ``face_detected``  (bool, optional) — whether a face was detected
            - ``face_similarity`` (float | None, optional) — face cosine similarity
        Additional fields are silently ignored.

    Returns
    -------
    dict
        Keys are track IDs (preserving the original type — int or str).
        Values are dicts with these keys:

        ``max_similarity``        float — highest single-crop body similarity
        ``mean_similarity``       float — mean over all body crops for the track
        ``mean_top3_similarity``  float — mean of the top-3 body crops
        ``num_observations``      int   — number of contributing crops
        ``face_coverage``         float — fraction of crops with face_detected=True
        ``face_primary``          bool  — True when face evidence is sufficient
                                          to use face-only identity for this track

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
    - face_coverage and face_primary default to 0.0 / False when no
      face fields are present in the records (body-only mode).
    """
    if not isinstance(records, list):
        raise TypeError(
            f"records must be a list, got {type(records).__name__}"
        )

    # Validate and group ------------------------------------------------
    grouped: dict[Any, list[float]] = defaultdict(list)
    # Face evidence per track: list of (face_sim, face_detected)
    face_evidence: dict[Any, list[tuple[float | None, bool]]] = defaultdict(list)

    for i, rec in enumerate(records):
        if "track_id" not in rec:
            raise ValueError(f"Record at index {i} is missing 'track_id'")
        if "similarity" not in rec:
            raise ValueError(f"Record at index {i} is missing 'similarity'")
        track_id = rec["track_id"]
        similarity = float(rec["similarity"])  # normalise to plain float
        grouped[track_id].append(similarity)

        # Collect face evidence if present in record
        face_sim = rec.get("face_similarity")  # may be None
        face_detected = rec.get("face_detected", False)
        face_evidence[track_id].append((face_sim, face_detected))

    # Empty input: return empty dict ------------------------------------
    if not grouped:
        return {}

    # Compute per-track face coverage
    face_coverage: dict[Any, float] = {}
    for tid, evidence in face_evidence.items():
        if evidence:
            face_coverage[tid] = sum(1 for _, det in evidence if det) / len(evidence)
        else:
            face_coverage[tid] = 0.0

    # Determine whether each track has sufficient reliable face evidence
    # Sufficient = face_coverage >= threshold AND at least one valid face_sim
    track_face_primary: dict[Any, bool] = {}
    for tid, evidence in face_evidence.items():
        has_valid_face = any(s is not None for s, _ in evidence)
        track_face_primary[tid] = (face_coverage[tid] >= _FACE_COVERAGE_MIN) and has_valid_face

    # Compute statistics ------------------------------------------------
    result: dict[Any, dict[str, float | int | bool]] = {}

    for track_id, sims in grouped.items():
        n = len(sims)

        # max
        max_sim = max(sims)

        # mean over all
        mean_sim = sum(sims) / n

        # mean of top-3 (or all if fewer than 3)
        sorted_desc = sorted(sims, reverse=True)
        top_k = sorted_desc[:3]  # slice gives <= 3 elements
        mean_top3 = sum(top_k) / len(top_k)

        result[track_id] = {
            "max_similarity": float(max_sim),
            "mean_similarity": float(mean_sim),
            "mean_top3_similarity": float(mean_top3),
            "num_observations": int(n),
            # Face-first fields
            "face_coverage": float(face_coverage.get(track_id, 0.0)),
            "face_primary": bool(track_face_primary.get(track_id, False)),
        }

    return result


def resolve_track_identity(
    track_stats: dict,
    face_sim: float | None,
    body_sim: float,
) -> tuple[str, float]:
    """
    Resolve the identity score for one track using the face-first policy.

    If track_stats indicates face_primary is True and face_sim is valid,
    returns ("face", face_sim).  Otherwise returns ("body", body_sim).

    This is the per-crop decision used by MatchingService, and it is
    deterministic: same inputs always produce same (mode, score).

    Parameters
    ----------
    track_stats:
        Per-track stats dict from aggregate_by_track(), must contain
        ``face_primary`` (bool).
    face_sim:
        Face cosine similarity for this crop, or None when no face
        was detected / embedding is invalid.
    body_sim:
        Body cosine similarity for this crop.

    Returns
    -------
    (matching_mode, identity_score)
        matching_mode: "face" or "body"
        identity_score: the selected similarity value (never a blend)
    """
    if track_stats.get("face_primary") and face_sim is not None:
        try:
            return "face", float(face_sim)
        except (TypeError, ValueError):
            # Non-numeric face_sim despite face_primary=True � safe body fallback
            pass
    return "body", float(body_sim)