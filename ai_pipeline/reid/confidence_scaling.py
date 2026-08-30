"""
Phase 5.4 — Similarity to Confidence Scaling

This module converts raw cosine similarity values to a scaled confidence score
in the range [0, 100] using min-max normalization based on observed Phase 4.5
similarity distributions.

IMPORTANT TERMINOLOGY:
This produces a "similarity-derived confidence score" or "scaled confidence score".
It is NOT a calibrated probability of identity and should NOT be interpreted as
"P(person is the same) = X%".

The scaling has not been statistically calibrated against ground-truth identity data.
It is a simple normalization to make similarity scores more interpretable for
human review and ranking.
"""

from typing import Union

# Phase 4.5 observed similarity range from dataset/reid_sanity/similarity_report.json
# Source: Same-track vs different-track cosine similarity distributions on C01 embeddings
# Total pairs: 642 same-track, 642 different-track
OBSERVED_MIN = 0.3315049352393543  # Minimum across all pairs (different-track min)
OBSERVED_MAX = 0.9730031552165505  # Maximum across all pairs (same-track max)


def similarity_to_confidence(
    similarity: Union[float, int],
    observed_min: float = OBSERVED_MIN,
    observed_max: float = OBSERVED_MAX,
) -> float:
    """
    Convert a cosine similarity value to a scaled confidence score in [0, 100].

    Formula:
        confidence = ((similarity - observed_min) /
                     (observed_max - observed_min)) * 100

    The result is clamped to the range [0, 100] to handle edge cases where
    similarity falls outside the observed range.

    Args:
        similarity: Raw cosine similarity value (typically in [-1, 1]).
        observed_min: Minimum observed similarity from Phase 4.5 sanity check.
                      Defaults to the actual measured value (0.3315...).
        observed_max: Maximum observed similarity from Phase 4.5 sanity check.
                      Defaults to the actual measured value (0.9730...).

    Returns:
        Scaled confidence score in [0, 100].

    Raises:
        ValueError: If observed_max equals observed_min (division by zero).

    Edge Cases:
        - If similarity < observed_min: returns 0.0 (clamped)
        - If similarity > observed_max: returns 100.0 (clamped)
        - If similarity == observed_min: returns 0.0
        - If similarity == observed_max: returns 100.0

    Examples:
        >>> similarity_to_confidence(0.8148)
        71.932...
        >>> similarity_to_confidence(0.8878)
        86.846...
        >>> similarity_to_confidence(0.3315049352393543)  # observed_min
        0.0
        >>> similarity_to_confidence(0.9730031552165505)  # observed_max
        100.0
    """
    if observed_max == observed_min:
        raise ValueError(
            f"observed_max ({observed_max}) equals observed_min ({observed_min}), "
            "cannot compute scaling factor (division by zero). "
            "This indicates a degenerate similarity distribution."
        )

    # Apply min-max scaling
    confidence = ((similarity - observed_min) / (observed_max - observed_min)) * 100

    # Clamp to [0, 100]
    confidence = max(0.0, min(100.0, confidence))

    return confidence


def get_observed_range() -> tuple[float, float]:
    """
    Return the Phase 4.5 observed similarity range.

    Returns:
        Tuple of (observed_min, observed_max) from the Phase 4.5 sanity check.
    """
    return OBSERVED_MIN, OBSERVED_MAX
