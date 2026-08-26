"""
quality_filter.py — Phase 3.5
Re-ID crop quality filtering for TRACE.

Evaluates whether a bounding box / crop is suitable for Re-ID embedding
generation BEFORE the crop image is saved or consumed downstream.

Two filter rules are applied in order:

  1. Minimum size check
     Reject if the clipped crop width < min_crop_width
                OR clipped crop height < min_crop_height.
     These thresholds are set in config.yaml under reid.min_crop_width /
     reid.min_crop_height and were raised in Phase 3.5 from 30×60 → 40×100.

  2. Severe edge-truncation check  (optional, controlled by config)
     Reject when ALL of the following are true:
       a. The bbox touches or is within edge_margin_pixels of a frame boundary
          (left, right, top, or bottom edge).
       b. The truncated dimension is "severe" — the crop is cut to less than
          half of the frame in that axis.
     Rationale: people legitimately enter/exit frame at edges and their crops
     are still useful.  We only reject when the bbox is hugging an edge AND
     the person is clearly heavily cropped (< 50 % of frame dimension in that
     axis), meaning very little of the body is visible.

Public API
----------
    is_crop_large_enough(crop_w, crop_h, min_w, min_h)  -> bool
    is_severely_truncated(bbox, img_w, img_h, margin)    -> bool
    evaluate_crop_quality(bbox, img_w, img_h, cfg)       -> dict

Return structure of evaluate_crop_quality
------------------------------------------
{
    "accepted": bool,
    "reason":   "accepted" | "invalid_bbox" | "crop_too_small" | "severely_truncated",
    "details":  str   # human-readable explanation, always present
}

This module does NOT read the video, run YOLO, or import OpenCV.
It only performs arithmetic on bbox coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_quality_config(config_path: Path = CONFIG_PATH) -> dict:
    """
    Return the reid config block from config.yaml with all Phase 3.5 keys.

    Fills in safe defaults for any key that is absent so older configs that
    predate Phase 3.5 still work without crashing.

    Returns a flat-ish dict:
        {
            "min_crop_width":                  int,
            "min_crop_height":                 int,
            "reject_severe_edge_truncation":   bool,
            "edge_margin_pixels":              int,
        }
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "reid" not in cfg:
        raise KeyError(f"'reid' section missing in {config_path}")

    reid = cfg["reid"]
    quality = reid.get("quality", {}) or {}

    return {
        "min_crop_width":                int(reid.get("min_crop_width",  40)),
        "min_crop_height":               int(reid.get("min_crop_height", 100)),
        "reject_severe_edge_truncation": bool(quality.get("reject_severe_edge_truncation", True)),
        "edge_margin_pixels":            int(quality.get("edge_margin_pixels", 5)),
    }


# ---------------------------------------------------------------------------
# Rule 1 — minimum size
# ---------------------------------------------------------------------------

def is_crop_large_enough(
    crop_w: int,
    crop_h: int,
    min_w: int,
    min_h: int,
) -> bool:
    """
    Return True if the crop meets minimum size requirements.

    Args:
        crop_w: Width of the (already-clipped) crop in pixels.
        crop_h: Height of the (already-clipped) crop in pixels.
        min_w:  Minimum required width (reid.min_crop_width in config).
        min_h:  Minimum required height (reid.min_crop_height in config).
    """
    return crop_w >= min_w and crop_h >= min_h


# ---------------------------------------------------------------------------
# Rule 2 — severe edge truncation
# ---------------------------------------------------------------------------

def is_severely_truncated(
    bbox: list | tuple,
    img_w: int,
    img_h: int,
    margin: int,
) -> bool:
    """
    Return True if the bbox indicates the person is severely truncated at a
    frame edge.

    The rule:
      For each of the four edges (left, right, top, bottom):
        - Does the bbox touch the edge within `margin` pixels?
        - If yes, is the crop dimension in that axis less than half the
          corresponding frame dimension?
      If both conditions hold for ANY single edge → severely truncated.

    This intentionally accepts people who merely walk into frame (they touch
    the edge but occupy a reasonable fraction of the frame height/width).
    It only rejects bboxes that are simultaneously edge-touching AND tiny in
    that axis — which means most of the person is off-screen.

    Args:
        bbox:   [x1, y1, x2, y2] — may be floats; clipping is the caller's
                responsibility before this check.
        img_w:  Frame width in pixels.
        img_h:  Frame height in pixels.
        margin: Pixel margin for "touching the edge" (edge_margin_pixels in
                config).

    Returns:
        True if severely truncated, False otherwise.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    crop_w = x2 - x1
    crop_h = y2 - y1

    half_w = img_w / 2.0
    half_h = img_h / 2.0

    # Left edge: x1 is within margin of 0, AND the crop is less than half the frame width
    if x1 <= margin and crop_w < half_w:
        return True

    # Right edge: x2 is within margin of img_w, AND the crop is less than half the frame width
    if x2 >= (img_w - margin) and crop_w < half_w:
        return True

    # Top edge: y1 is within margin of 0, AND the crop is less than half the frame height
    if y1 <= margin and crop_h < half_h:
        return True

    # Bottom edge: y2 is within margin of img_h, AND the crop is less than half the frame height
    if y2 >= (img_h - margin) and crop_h < half_h:
        return True

    return False


# ---------------------------------------------------------------------------
# Combined evaluator — single call site for crop_extractor.py
# ---------------------------------------------------------------------------

def evaluate_crop_quality(
    bbox: list | tuple,
    img_w: int,
    img_h: int,
    cfg: dict,
) -> dict[str, Any]:
    """
    Evaluate whether a crop should be accepted for Re-ID.

    Applies rules in a defined order and returns on the first rejection so
    that each rejected crop has exactly one explicit reason.

    Args:
        bbox:   Raw (pre-clip) or clipped bbox [x1, y1, x2, y2].
                This function clips internally for the size check so the
                caller does not need to pre-clip, but the clipped coordinates
                are what matter for the result.
        img_w:  Frame width in pixels.
        img_h:  Frame height in pixels.
        cfg:    Quality config dict as returned by load_quality_config().
                Required keys:
                    min_crop_width, min_crop_height,
                    reject_severe_edge_truncation, edge_margin_pixels

    Returns:
        {
            "accepted": bool,
            "reason":   str,   # see module docstring for possible values
            "details":  str,
        }
    """
    min_w  = int(cfg["min_crop_width"])
    min_h  = int(cfg["min_crop_height"])
    margin = int(cfg["edge_margin_pixels"])
    check_truncation = bool(cfg["reject_severe_edge_truncation"])

    # --- Rule 0: bbox sanity ---
    if bbox is None or len(bbox) != 4:
        return {
            "accepted": False,
            "reason":   "invalid_bbox",
            "details":  f"bbox is None or does not have 4 coordinates: {bbox}",
        }

    # Clip coordinates to frame for the size calculation
    x1 = max(0, min(int(bbox[0]), img_w))
    y1 = max(0, min(int(bbox[1]), img_h))
    x2 = max(0, min(int(bbox[2]), img_w))
    y2 = max(0, min(int(bbox[3]), img_h))

    if x2 <= x1 or y2 <= y1:
        return {
            "accepted": False,
            "reason":   "invalid_bbox",
            "details":  (
                f"bbox {list(bbox)} degenerates after clipping to frame "
                f"{img_w}×{img_h}: clipped=({x1},{y1},{x2},{y2})"
            ),
        }

    crop_w = x2 - x1
    crop_h = y2 - y1

    # --- Rule 1: minimum size ---
    if not is_crop_large_enough(crop_w, crop_h, min_w, min_h):
        parts = []
        if crop_w < min_w:
            parts.append(f"crop width {crop_w} < minimum {min_w}")
        if crop_h < min_h:
            parts.append(f"crop height {crop_h} < minimum {min_h}")
        return {
            "accepted": False,
            "reason":   "crop_too_small",
            "details":  f"crop size {crop_w}×{crop_h} is below minimum {min_w}×{min_h} — " + "; ".join(parts),
        }

    # --- Rule 2: severe edge truncation (optional) ---
    if check_truncation and is_severely_truncated(
        (x1, y1, x2, y2), img_w, img_h, margin
    ):
        # Identify which edge(s) triggered the rejection for the details string
        edges = []
        half_w = img_w / 2.0
        half_h = img_h / 2.0
        if x1 <= margin and crop_w < half_w:
            edges.append(f"left (x1={x1}≤{margin}, crop_w={crop_w}<{half_w:.0f})")
        if x2 >= (img_w - margin) and crop_w < half_w:
            edges.append(f"right (x2={x2}≥{img_w - margin}, crop_w={crop_w}<{half_w:.0f})")
        if y1 <= margin and crop_h < half_h:
            edges.append(f"top (y1={y1}≤{margin}, crop_h={crop_h}<{half_h:.0f})")
        if y2 >= (img_h - margin) and crop_h < half_h:
            edges.append(f"bottom (y2={y2}≥{img_h - margin}, crop_h={crop_h}<{half_h:.0f})")
        return {
            "accepted": False,
            "reason":   "severely_truncated",
            "details":  (
                f"crop {crop_w}×{crop_h} is severely truncated at edge(s): "
                + "; ".join(edges)
            ),
        }

    # --- All checks passed ---
    return {
        "accepted": True,
        "reason":   "accepted",
        "details":  f"crop size {crop_w}×{crop_h} meets all quality thresholds",
    }
