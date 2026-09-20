"""
kpr_similarity.py — Part-aware similarity utilities for TRACE Re-ID
====================================================================
Parallel to similarity.py (cosine_similarity) and track_aggregation.py
(aggregate_by_track), but operates on the structured KPR output produced
by embed_kpr.py: per-part embeddings, per-part visibility scores, and an
optional holistic embedding as fallback.

Public API
----------
    part_aware_similarity(emb_a, vis_a, emb_b, vis_b, ...)  -> float
    aggregate_kpr_by_track(kpr_records, query_kpr)          -> dict

Design decisions
----------------
- Only mutually visible parts (both vis_a[i] and vis_b[i] >= vis_threshold)
  contribute to the similarity.  This is the core advantage of KPR for
  partial / occluded bodies — comparing a waist-up crop against a full-body
  crop only uses the torso/arms/head parts, not the invisible legs.

- Weighted mean: each visible part contributes with weight = min(vis_a[i],
  vis_b[i]).  The minimum is the conservative "confidence" that both images
  show this part.  This penalises parts where one image barely meets the
  threshold (e.g., a partially cropped shoulder at 0.31) versus a fully
  visible part (0.95 × 0.92).

- Holistic fallback: when NO parts are mutually visible (e.g. two completely
  disjoint crops of the same person), part_aware_similarity falls back to
  cosine similarity of the holistic embeddings.  Returns 0.0 if holistic is
  also absent.

- aggregate_kpr_by_track() mirrors aggregate_by_track() from
  track_aggregation.py: it operates on crop-level KPR records, computes a
  part-aware similarity between each gallery crop and the query KPR record,
  then aggregates to per-track max / mean_top3 / mean_visible_parts.

- All return values are plain Python float / int — directly JSON-serialisable.
  No NumPy scalar types leak out of this module.

- Pure NumPy / pure Python — no GPU, no torchreid, no ONNX.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional, Sequence, Union

import numpy as np

from ai_pipeline.reid.similarity import cosine_similarity   # reuse existing, tested impl

# Type alias
_VectorLike = Union[Sequence[float], np.ndarray]

# Default visibility threshold — parts below this are treated as invisible
_DEFAULT_VIS_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# part_aware_similarity
# ---------------------------------------------------------------------------

def part_aware_similarity(
    emb_a: list[list[float]],
    vis_a: list[float],
    emb_b: list[list[float]],
    vis_b: list[float],
    holistic_a: Optional[list[float]] = None,
    holistic_b: Optional[list[float]] = None,
    vis_threshold: float = _DEFAULT_VIS_THRESHOLD,
) -> float:
    """
    Compute part-aware cosine similarity between two KPR crop embeddings.

    For each body part i, the part contributes to the score only when BOTH
    vis_a[i] >= vis_threshold AND vis_b[i] >= vis_threshold.
    The weight of each included part is min(vis_a[i], vis_b[i]).

    Falls back to holistic cosine similarity when no parts are mutually
    visible.  Returns 0.0 if both the part and holistic fallbacks fail.

    Args:
        emb_a:         Per-part embeddings for sample A.
                       Shape: list of num_parts vectors, each D-dim.
        vis_a:         Per-part visibility scores for sample A.
                       Length: num_parts, each in [0, 1].
        emb_b:         Per-part embeddings for sample B (same num_parts, D).
        vis_b:         Per-part visibility scores for sample B.
        holistic_a:    Holistic (global) embedding for sample A, or None.
        holistic_b:    Holistic (global) embedding for sample B, or None.
        vis_threshold: Minimum visibility for a part to be included.
                       Default 0.3 (matches config kpr_vis_threshold).

    Returns:
        Float in [-1.0, 1.0].
        - Weighted mean of visible-part cosine similarities when parts exist.
        - Holistic cosine similarity as fallback.
        - 0.0 if no usable signal is available.

    Raises:
        ValueError: if emb_a and emb_b have different numbers of parts,
                    or if vis_a / vis_b lengths do not match the embeddings.

    Examples:
        >>> # Two fully visible crops — all 5 parts used
        >>> sim = part_aware_similarity(
        ...     emb_a=[[1,0],[0,1],[1,0],[0,1],[1,0]],
        ...     vis_a=[0.9, 0.8, 0.9, 0.7, 0.8],
        ...     emb_b=[[1,0],[0,1],[1,0],[0,1],[1,0]],
        ...     vis_b=[0.8, 0.9, 0.7, 0.9, 0.9],
        ... )
        >>> sim
        1.0

        >>> # Waist-up crop A (legs invisible) vs full-body crop B
        >>> # Parts 3 and 4 (legs) are below threshold in A — excluded
        >>> sim = part_aware_similarity(
        ...     emb_a=[[1,0],[0,1],[1,0],[0,0],[0,0]],
        ...     vis_a=[0.9, 0.8, 0.7, 0.1, 0.05],
        ...     emb_b=[[1,0],[0,1],[1,0],[0,1],[0,1]],
        ...     vis_b=[0.8, 0.9, 0.7, 0.8,  0.9],
        ... )
        >>> # Only parts 0, 1, 2 contribute — correct result
        >>> sim
        1.0
    """
    num_parts = len(emb_a)

    # Input validation
    if len(emb_b) != num_parts:
        raise ValueError(
            f"Part count mismatch: emb_a has {num_parts} parts, "
            f"emb_b has {len(emb_b)} parts."
        )
    if len(vis_a) != num_parts:
        raise ValueError(
            f"vis_a length {len(vis_a)} does not match num_parts {num_parts}."
        )
    if len(vis_b) != num_parts:
        raise ValueError(
            f"vis_b length {len(vis_b)} does not match num_parts {num_parts}."
        )

    weighted_sum = 0.0
    weight_total = 0.0

    for i in range(num_parts):
        va = float(vis_a[i])
        vb = float(vis_b[i])

        # Only use parts that are visible in BOTH images
        if va < vis_threshold or vb < vis_threshold:
            continue

        try:
            sim = cosine_similarity(emb_a[i], emb_b[i])
        except (ValueError, Exception):
            # Dimension mismatch or empty vector — skip this part silently
            continue

        weight = min(va, vb)     # conservative: trust the less-visible view
        weighted_sum += weight * sim
        weight_total += weight

    if weight_total > 0.0:
        return float(weighted_sum / weight_total)

    # --- Holistic fallback ------------------------------------------------
    # No mutually visible parts — fall back to global embedding comparison.
    if holistic_a is not None and holistic_b is not None:
        try:
            return cosine_similarity(holistic_a, holistic_b)
        except (ValueError, Exception):
            pass

    # Nothing usable — return neutral score
    return 0.0


# ---------------------------------------------------------------------------
# aggregate_kpr_by_track
# ---------------------------------------------------------------------------

def aggregate_kpr_by_track(
    kpr_records: list[dict[str, Any]],
    query_kpr: dict[str, Any],
    vis_threshold: float = _DEFAULT_VIS_THRESHOLD,
) -> dict[int | str, dict[str, float | int]]:
    """
    Compute per-track KPR aggregation statistics against a query crop.

    For each gallery crop record, computes part_aware_similarity() against
    the query_kpr record.  Then aggregates per track into:

        max_part_aware_sim       — highest single-crop similarity
        mean_top3_part_aware_sim — mean of the top-3 crop similarities
                                   (or all if fewer than 3 exist)
        mean_visible_parts       — average number of mutually visible parts
                                   across all crops for this track
        num_observations         — number of gallery crops for the track

    Parameters
    ----------
    kpr_records:
        Gallery KPR records (list of dicts from kpr_embeddings_<cam>.json).
        Each record must contain:
            - ``track_id``        (int or str)
            - ``part_embeddings`` (list of lists)
            - ``part_visibility`` (list of floats)
        Optionally:
            - ``holistic_embedding`` (list of floats) — used as fallback

    query_kpr:
        A single KPR record for the query crop (same schema as gallery).
        Must contain ``part_embeddings`` and ``part_visibility``.
        May contain ``holistic_embedding``.

    vis_threshold:
        Passed through to part_aware_similarity(). Default 0.3.

    Returns
    -------
    dict
        Keys are track IDs (preserving original type — int or str).
        Values are dicts with exactly:

        ``max_part_aware_sim``       float
        ``mean_top3_part_aware_sim`` float
        ``mean_visible_parts``       float
        ``num_observations``         int

    Raises
    ------
    TypeError
        If ``kpr_records`` is not a list.
    ValueError
        If any record is missing required fields.
    KeyError
        If ``query_kpr`` is missing ``part_embeddings`` or ``part_visibility``.

    Notes
    -----
    - Empty input returns an empty dict without raising.
    - All returned numeric values are plain Python float / int.
    """
    if not isinstance(kpr_records, list):
        raise TypeError(
            f"kpr_records must be a list, got {type(kpr_records).__name__}"
        )

    # Extract query fields once (fail fast if malformed)
    q_part_embs = query_kpr["part_embeddings"]
    q_part_vis  = query_kpr["part_visibility"]
    q_holistic  = query_kpr.get("holistic_embedding")

    # Validate and group gallery records by track_id
    grouped: dict[Any, list[dict]] = defaultdict(list)
    for idx, rec in enumerate(kpr_records):
        if "track_id" not in rec:
            raise ValueError(f"Record at index {idx} is missing 'track_id'")
        if "part_embeddings" not in rec:
            raise ValueError(f"Record at index {idx} is missing 'part_embeddings'")
        if "part_visibility" not in rec:
            raise ValueError(f"Record at index {idx} is missing 'part_visibility'")
        grouped[rec["track_id"]].append(rec)

    if not grouped:
        return {}

    result: dict[Any, dict[str, float | int]] = {}

    for track_id, crops in grouped.items():
        sims:           list[float] = []
        visible_counts: list[int]   = []

        for rec in crops:
            g_part_embs = rec["part_embeddings"]
            g_part_vis  = rec["part_visibility"]
            g_holistic  = rec.get("holistic_embedding")

            # Count mutually visible parts for this crop pair
            n_visible = sum(
                1 for i in range(min(len(q_part_vis), len(g_part_vis)))
                if q_part_vis[i] >= vis_threshold and g_part_vis[i] >= vis_threshold
            )
            visible_counts.append(n_visible)

            try:
                sim = part_aware_similarity(
                    emb_a=q_part_embs,
                    vis_a=q_part_vis,
                    emb_b=g_part_embs,
                    vis_b=g_part_vis,
                    holistic_a=q_holistic,
                    holistic_b=g_holistic,
                    vis_threshold=vis_threshold,
                )
            except (ValueError, Exception):
                # Mismatched part counts between query and this gallery crop —
                # fall back to holistic if available, else skip
                if q_holistic is not None and g_holistic is not None:
                    try:
                        sim = cosine_similarity(q_holistic, g_holistic)
                    except Exception:
                        continue
                else:
                    continue

            sims.append(sim)

        if not sims:
            # All crops for this track were unusable
            result[track_id] = {
                "max_part_aware_sim":       0.0,
                "mean_top3_part_aware_sim": 0.0,
                "mean_visible_parts":       0.0,
                "num_observations":         int(len(crops)),
            }
            continue

        n = len(sims)
        max_sim  = max(sims)

        sorted_desc = sorted(sims, reverse=True)
        top_k       = sorted_desc[:3]
        mean_top3   = sum(top_k) / len(top_k)

        mean_vis_parts = (
            float(sum(visible_counts)) / float(len(visible_counts))
            if visible_counts else 0.0
        )

        result[track_id] = {
            "max_part_aware_sim":       float(max_sim),
            "mean_top3_part_aware_sim": float(mean_top3),
            "mean_visible_parts":       float(mean_vis_parts),
            "num_observations":         int(n),
        }

    return result
