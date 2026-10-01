"""
matching.py — Face-first identity matching for TRACE.
================================================================
Replaces the old triple-fusion formula with a deterministic
conditional policy:

    IF a reliable face embedding is available for a crop/track:
        matching_mode = "face"
        identity_score = face_similarity ONLY

    ELSE:
        matching_mode = "body"
        identity_score = body/part Re-ID score (existing fallback)

This is NOT a weighted fusion.  Face mode never blends OSNet,
KPR, or clothing/body similarity into the identity score.

Public API
----------
    select_identity_score(face_sim, body_sim, matching_mode) -> float
    resolve_match_mode(face_sim, face_coverage) -> str

Design decisions
----------------
- Face usability is governed by the same face_coverage >= 0.3 rule
  already enforced inside MatchingService._compute_face_similarities().
  When that method returns face_sim=None the crop/track has no
  reliable face signal and the body fallback is used automatically.
- A valid face embedding (non-None face_sim) is sufficient to select
  face mode.  No separate "confidence boost" or weight is applied.
- KPR / SOLIDER are never consulted in face mode — they are body-
  only signals and are silently ignored for that crop.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Face usability threshold (mirrors matching_service._FACE_COVERAGE_MIN)
# ---------------------------------------------------------------------------
_FACE_COVERAGE_MIN: float = 0.3


def resolve_match_mode(face_sim, face_coverage: float = 1.0) -> str:
    """
    Return "face" or "body" for a single crop/track.

    Args:
        face_sim:        Face cosine similarity, or None when no face
                         was detected / embedding is invalid / coverage
                         is below threshold.
        face_coverage:   Fraction of crops in the track with a detected
                         face (0.0–1.0).  Defaults to 1.0 for a single
                         crop that already passed coverage filtering.

    Returns:
        "face" when face_sim is a valid float and coverage is sufficient.
        "body" otherwise.
    """
    if face_sim is None:
        return "body"
    try:
        sim_val = float(face_sim)
    except (TypeError, ValueError):
        return "body"
    if not (-1.0 <= sim_val <= 1.0):
        return "body"
    if float(face_coverage) < _FACE_COVERAGE_MIN:
        return "body"
    return "face"


def select_identity_score(
    face_sim,
    body_sim: float,
    matching_mode: str = "body",
) -> float:
    """
    Return the identity score according to the face-first policy.

    Args:
        face_sim:       Face cosine similarity, or None.
        body_sim:       Body Re-ID cosine similarity.
        matching_mode:  "face" or "body" as returned by resolve_match_mode().

    Returns:
        identity_score in the same [0, 1] similarity space as the inputs.
        Never converts similarity into a fake probability.
    """
    if matching_mode == "face":
        # Face mode: identity_score = face similarity ONLY.
        # body_sim and any KPR/SOLIDER signal are explicitly ignored.
        try:
            return float(face_sim)
        except (TypeError, ValueError):
            # Invalid face_sim despite mode="face" — safe fallback
            return float(body_sim)
    # Body mode: use existing body/part Re-ID score unchanged.
    return float(body_sim)
