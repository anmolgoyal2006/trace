"""
confidence_scaling.py — Backbone-Aware Similarity-to-Confidence Scaling
=======================================================================

Converts raw cosine similarity to a scaled display score in [0, 100].

CRITICAL TERMINOLOGY — read before using
-----------------------------------------
The output of similarity_to_confidence() is a **similarity-derived confidence
score** (also called "match score" in the UI).  It is NOT:

  - A calibrated probability of identity  (P(same person) = X%)
  - An accuracy percentage
  - A statistically validated measure

It is a simple min-max rescaling of cosine similarity into a human-readable
0–100 range, using the *observed* similarity distribution from Phase 4.5
measurements on the TRACE C01 gallery.  The mapping is monotonic and useful
for ranking candidates, but must NOT be interpreted as "86% chance this is
the same person."

Proper calibration (Platt scaling, isotonic regression, etc.) requires a
labelled genuine/impostor dataset that does not currently exist in this repo.

Backbone-specific ranges
------------------------
OSNet x1_0 (512-dim) and SOLIDER Swin-Small (768-dim) have different cosine
similarity distributions.  Using OSNet calibration numbers to scale a SOLIDER
similarity would produce systematically wrong display scores.

Observed distributions (Phase 4.5, C01 gallery, OSNet):
  min across all pairs : 0.3315  (different-track minimum)
  max across all pairs : 0.9730  (same-track maximum)

SOLIDER distribution characteristics (from cross-track analysis):
  different-track minimum ≈ 0.83
  different-track mean    ≈ 0.92
  >0.85 for ≈99% of different-track pairs
  → OSNet threshold 0.74 is completely invalid for SOLIDER
  → Initial empirical threshold 0.94 (NOT statistically calibrated)

These SOLIDER values are INITIAL / EMPIRICAL starting points only.
They are NOT calibrated thresholds.  Proper calibration requires a labelled
genuine/impostor dataset.  Treat them as:
  "experimentally observed — adjust after validation with ground truth data"

Public API
----------
    similarity_to_confidence(sim, embedding_dim)  → float   [0, 100]
    get_backbone_config(embedding_dim)             → BackboneConfig
    BACKBONE_OSNET_512                             BackboneConfig constant
    BACKBONE_SOLIDER_768                           BackboneConfig constant
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# ---------------------------------------------------------------------------
# BackboneConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackboneConfig:
    """
    Per-backbone similarity scaling configuration.

    All fields are empirical observations, NOT statistically calibrated
    probabilities.  The labels below describe their intended usage.

    Attributes
    ----------
    name
        Human-readable backbone identifier, used in logs.
    embedding_dim
        Expected embedding vector dimension (used for backbone selection).
    observed_min
        Lower bound of the observed cosine similarity range.
        Similarities at or below this map to 0% display score.
    observed_max
        Upper bound of the observed cosine similarity range.
        Similarities at or above this map to 100% display score.
    no_match_threshold
        Minimum cosine similarity (or fused score) for a track to be
        considered a potential match.  Below this → NO_CONFIDENT_MATCH.
        EMPIRICAL / MVP value — not a calibrated decision boundary.
    soft_floor
        Similarities above soft_floor but below no_match_threshold are
        returned as POSSIBLE_MATCH candidates for human review.
        Below soft_floor → not returned at all.
        EMPIRICAL / MVP value.
    """
    name: str
    embedding_dim: int
    observed_min: float
    observed_max: float
    no_match_threshold: float   # empirical — see module docstring
    soft_floor: float           # empirical — for POSSIBLE_MATCH candidates


# ---------------------------------------------------------------------------
# Backbone configurations
# ---------------------------------------------------------------------------

BACKBONE_OSNET_512 = BackboneConfig(
    name="OSNet x1_0",
    embedding_dim=512,
    # Phase 4.5 measurements on C01 gallery (642 same-track + 642 diff-track pairs)
    observed_min=0.3315049352393543,
    observed_max=0.9730031552165505,
    # Phase 5.7 evidence-based threshold (small validation set only):
    #   rejects unknown (sim=0.7267), retains known matches (>=0.7974)
    # NOT a universally calibrated threshold.
    no_match_threshold=0.74,
    soft_floor=0.60,
)

BACKBONE_SOLIDER_768 = BackboneConfig(
    name="SOLIDER Swin-Small",
    embedding_dim=768,
    # INITIAL EMPIRICAL VALUES from cross-track analysis.
    # Different-track pairs cluster around 0.83-0.92.
    # These are NOT statistically calibrated thresholds.
    # Recalibrate with labelled genuine/impostor data before production use.
    observed_min=0.83,
    observed_max=0.995,
    # Initial empirical threshold: 0.94.
    # Rationale: >99% of different-track pairs are above 0.85, so 0.74 is
    # useless; 0.94 is an initial conservative estimate only.
    # MUST be re-validated with ground-truth identity labels.
    no_match_threshold=0.94,
    soft_floor=0.90,
)

# Lookup by embedding dimension
_BACKBONE_BY_DIM: dict[int, BackboneConfig] = {
    512: BACKBONE_OSNET_512,
    768: BACKBONE_SOLIDER_768,
}

# Legacy constants preserved for backwards compatibility with existing tests
OBSERVED_MIN = BACKBONE_OSNET_512.observed_min
OBSERVED_MAX = BACKBONE_OSNET_512.observed_max


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_backbone_config(embedding_dim: int) -> BackboneConfig:
    """
    Return the BackboneConfig for the given embedding dimension.

    Falls back to OSNet (512) for unknown dimensions and logs a warning.
    Never raises — the caller should not fail due to an unknown backbone.

    Args:
        embedding_dim: Dimension of the embedding vector (e.g. 512 or 768).

    Returns:
        Matching BackboneConfig, defaulting to BACKBONE_OSNET_512 when unknown.
    """
    cfg = _BACKBONE_BY_DIM.get(embedding_dim)
    if cfg is None:
        # Unknown backbone — fall back to OSNet but log a warning.
        # Import loguru lazily to avoid a hard dependency in pure AI code.
        try:
            from loguru import logger
            logger.warning(
                f"[confidence_scaling] Unknown embedding_dim={embedding_dim}; "
                f"falling back to OSNet range. "
                f"This will produce incorrect confidence scores for non-OSNet backbones. "
                f"Add a BackboneConfig for dim={embedding_dim}."
            )
        except ImportError:
            pass
        return BACKBONE_OSNET_512
    return cfg


def similarity_to_confidence(
    similarity: Union[float, int],
    observed_min: float = OBSERVED_MIN,
    observed_max: float = OBSERVED_MAX,
    embedding_dim: int = 512,
) -> float:
    """
    Convert a cosine similarity value to a scaled display score in [0, 100].

    IMPORTANT: The returned value is a **similarity-derived match score**,
    NOT a calibrated identity probability.  Do not display it as "accuracy"
    or "P(same person)".  Label it "Match score" or "Similarity score" in
    the UI.

    Backbone selection:
        When embedding_dim=768 (SOLIDER), the SOLIDER observed range is used
        automatically via get_backbone_config().  The explicit observed_min /
        observed_max arguments override backbone auto-selection and should only
        be used in tests or for custom calibration.

    Formula:
        score = ((similarity - observed_min) / (observed_max - observed_min)) * 100
        clamped to [0, 100].

    Args:
        similarity:    Raw cosine similarity (typically in [-1, 1]).
        observed_min:  Override lower bound (default: OSNet Phase 4.5 value).
                       Pass the backbone's observed_min for correct scaling.
        observed_max:  Override upper bound (default: OSNet Phase 4.5 value).
        embedding_dim: Embedding dimension — used to auto-select backbone range
                       when observed_min/max are not explicitly overridden.
                       512 → OSNet range, 768 → SOLIDER range.

    Returns:
        Float in [0.0, 100.0].  This is a similarity-derived match score,
        NOT an identity probability.

    Raises:
        ValueError: if observed_max == observed_min (degenerate distribution).

    Examples:
        >>> # OSNet
        >>> similarity_to_confidence(0.8148, embedding_dim=512)
        75.36...  # NOT "75% chance it's the same person"

        >>> # SOLIDER — uses SOLIDER range automatically
        >>> similarity_to_confidence(0.89, embedding_dim=768)
        35.8...   # very different from OSNet result for same raw value

        >>> # OSNet range used explicitly
        >>> similarity_to_confidence(0.89, observed_min=0.3315, observed_max=0.9730)
        87.0...   # OSNet-scaled result
    """
    # Auto-select backbone range when the defaults haven't been overridden
    if observed_min == OBSERVED_MIN and observed_max == OBSERVED_MAX and embedding_dim != 512:
        cfg = get_backbone_config(embedding_dim)
        observed_min = cfg.observed_min
        observed_max = cfg.observed_max

    if observed_max == observed_min:
        raise ValueError(
            f"observed_max ({observed_max}) equals observed_min ({observed_min}), "
            "cannot compute scaling factor (division by zero). "
            "This indicates a degenerate similarity distribution."
        )

    score = ((similarity - observed_min) / (observed_max - observed_min)) * 100
    return float(max(0.0, min(100.0, score)))


def get_observed_range(embedding_dim: int = 512) -> tuple[float, float]:
    """
    Return the (observed_min, observed_max) for the given backbone.

    Args:
        embedding_dim: 512 for OSNet, 768 for SOLIDER.

    Returns:
        Tuple of (observed_min, observed_max).
    """
    cfg = get_backbone_config(embedding_dim)
    return cfg.observed_min, cfg.observed_max
