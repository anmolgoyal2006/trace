"""
similarity.py — Phase 4.5
Cosine similarity and statistical utilities for TRACE Re-ID sanity check.

This module is pure NumPy — it does NOT load torchreid, OSNet, or any GPU
resource. It operates on already-computed embedding vectors.

Public API
----------
    cosine_similarity(a, b)      -> float
    compute_stats(values)        -> dict
    threshold_fractions(values, thresholds) -> dict

Design decisions
----------------
- Accepts Python lists, NumPy arrays, or any sequence of numbers.
- Converts inputs to float64 for numerical stability.
- Returns plain Python float scalars (JSON-serialisable).
- Zero-vector guard: returns 0.0 when either input is the zero vector.
  This is a defined fallback, not a silent NaN.
- All statistics in compute_stats() are plain Python floats / ints so the
  caller can json.dump() the result directly.
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np

# Type alias for anything we can convert to a 1-D float array
_VectorLike = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: _VectorLike, b: _VectorLike) -> float:
    """
    Compute the cosine similarity between two vectors.

    Formula:
        cos(A, B) = dot(A, B) / (||A|| * ||B||)

    Args:
        a: First vector — Python list, NumPy array, or any numeric sequence.
        b: Second vector — must have the same length as *a*.

    Returns:
        Cosine similarity as a Python float in [-1.0, 1.0].
        Returns 0.0 if either vector is the zero vector.

    Raises:
        ValueError: if the vectors have different lengths or length 0.
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()

    if a_arr.size == 0 or b_arr.size == 0:
        raise ValueError("Vectors must be non-empty.")
    if a_arr.size != b_arr.size:
        raise ValueError(
            f"Vector length mismatch: {a_arr.size} vs {b_arr.size}."
        )

    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))

    # Guard: zero vector → cosine is undefined; return 0.0 as a safe fallback
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    raw = float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

    # Clamp to [-1, 1] to absorb floating-point rounding beyond these bounds
    return float(max(-1.0, min(1.0, raw)))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(values: Sequence[float]) -> dict:
    """
    Compute descriptive statistics for a sequence of scalar similarity values.

    Args:
        values: List (or array) of cosine similarity floats.

    Returns:
        Dict with keys:
            count   int
            mean    float
            median  float
            std     float
            min     float
            max     float
            p25     float   (25th percentile)
            p75     float   (75th percentile)

    Raises:
        ValueError: if *values* is empty.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute statistics on an empty sequence.")

    return {
        "count":  int(arr.size),
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr, ddof=0)),   # population std
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
    }


# ---------------------------------------------------------------------------
# Threshold diagnostics
# ---------------------------------------------------------------------------

def threshold_fractions(
    values: Sequence[float],
    thresholds: Sequence[float],
) -> dict[float, float]:
    """
    For each threshold, return the fraction of values that exceed it.

    Args:
        values:     Sequence of similarity floats.
        thresholds: Sequence of threshold values (e.g. [0.5, 0.6, 0.7, 0.8, 0.9]).

    Returns:
        Dict mapping each threshold -> fraction of values strictly above it,
        as a float in [0.0, 1.0].

    Raises:
        ValueError: if *values* is empty.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute threshold fractions on an empty sequence.")

    result: dict[float, float] = {}
    for t in thresholds:
        result[float(t)] = float(np.mean(arr > t))
    return result
