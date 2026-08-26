"""
test_quality_filter.py — Phase 3.5
Unit tests for quality_filter.py.

Tests cover:
  TEST 1  — valid crop 100×300 → accepted
  TEST 2  — width below threshold (39×150) → rejected
  TEST 3  — height below threshold (50×99) → rejected
  TEST 4  — exactly minimum dimensions (40×100) → accepted
  TEST 5  — invalid bbox → rejected
  TEST 6  — severe edge truncation enabled → rejected when rule satisfied
  TEST 7  — edge truncation disabled → same crop not rejected for that reason
  TEST 8  — configuration values loaded correctly from config.yaml
  TEST 9  — rejected crop always contains an explicit reason string
  TEST 10 — real C01 crops_metadata.json can be evaluated without crashing

Run with:
    python -m pytest ai_pipeline/reid/test_quality_filter.py -v
  or:
    python ai_pipeline/reid/test_quality_filter.py
"""

import json
import sys
from pathlib import Path

import pytest

# Allow direct import from the reid package directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quality_filter import (
    evaluate_crop_quality,
    is_crop_large_enough,
    is_severely_truncated,
    load_quality_config,
)

# ---------------------------------------------------------------------------
# Shared config for most tests — matches Phase 3.5 recommended values
# ---------------------------------------------------------------------------

BASE_CFG = {
    "min_crop_width":                 40,
    "min_crop_height":                100,
    "reject_severe_edge_truncation":  True,
    "edge_margin_pixels":             5,
}

# Frame dimensions used throughout (typical 1080p)
FRAME_W = 1920
FRAME_H = 1080


# ---------------------------------------------------------------------------
# TEST 1 — Valid crop well above minimum → accepted
# ---------------------------------------------------------------------------

def test_01_valid_crop_accepted():
    """A 100×300 crop centred in frame should be accepted."""
    # Place it well away from any edge
    bbox = [200, 100, 300, 400]   # width=100, height=300
    result = evaluate_crop_quality(bbox, FRAME_W, FRAME_H, BASE_CFG)
    assert result["accepted"] is True, f"Expected accepted, got: {result}"
    assert result["reason"] == "accepted"


# ---------------------------------------------------------------------------
# TEST 2 — Width below threshold → rejected
# ---------------------------------------------------------------------------

def test_02_width_below_threshold():
    """A 39×150 crop should be rejected for being too small (width < 40)."""
    # bbox that produces width=39, height=150 — far from any edge
    bbox = [200, 100, 239, 250]   # width=39, height=150
    result = evaluate_crop_quality(bbox, FRAME_W, FRAME_H, BASE_CFG)
    assert result["accepted"] is False
    assert result["reason"] == "crop_too_small"
    assert "width" in result["details"].lower()


# ---------------------------------------------------------------------------
# TEST 3 — Height below threshold → rejected
# ---------------------------------------------------------------------------

def test_03_height_below_threshold():
    """A 50×99 crop should be rejected for being too small (height < 100)."""
    bbox = [200, 100, 250, 199]   # width=50, height=99
    result = evaluate_crop_quality(bbox, FRAME_W, FRAME_H, BASE_CFG)
    assert result["accepted"] is False
    assert result["reason"] == "crop_too_small"
    assert "height" in result["details"].lower()


# ---------------------------------------------------------------------------
# TEST 4 — Exactly minimum dimensions → accepted
# ---------------------------------------------------------------------------

def test_04_exact_minimum_dimensions_accepted():
    """A 40×100 crop exactly at the minimum should be accepted (>= not >)."""
    bbox = [200, 100, 240, 200]   # width=40, height=100
    result = evaluate_crop_quality(bbox, FRAME_W, FRAME_H, BASE_CFG)
    assert result["accepted"] is True, f"Expected accepted at exact minimum, got: {result}"
    assert result["reason"] == "accepted"


# ---------------------------------------------------------------------------
# TEST 5 — Invalid bbox → rejected with reason invalid_bbox
# ---------------------------------------------------------------------------

def test_05_invalid_bbox_none():
    """None bbox should be rejected with reason invalid_bbox."""
    result = evaluate_crop_quality(None, FRAME_W, FRAME_H, BASE_CFG)
    assert result["accepted"] is False
    assert result["reason"] == "invalid_bbox"


def test_05b_invalid_bbox_degenerate():
    """Bbox that degenerates to zero area after clipping should be rejected."""
    # x1==x2 → zero width
    bbox = [100, 100, 100, 300]
    result = evaluate_crop_quality(bbox, FRAME_W, FRAME_H, BASE_CFG)
    assert result["accepted"] is False
    assert result["reason"] == "invalid_bbox"


def test_05c_invalid_bbox_wrong_length():
    """Bbox with wrong number of coordinates should be rejected."""
    bbox = [100, 100, 300]   # only 3 elements
    result = evaluate_crop_quality(bbox, FRAME_W, FRAME_H, BASE_CFG)
    assert result["accepted"] is False
    assert result["reason"] == "invalid_bbox"


# ---------------------------------------------------------------------------
# TEST 6 — Severe edge truncation enabled → rejected
# ---------------------------------------------------------------------------

def test_06_severe_truncation_enabled_rejected():
    """
    A crop that touches the left edge AND is less than half the frame width
    should be rejected when reject_severe_edge_truncation=True.

    Example:
      Frame: 1920×1080
      bbox: [0, 100, 200, 400]  → x1=0 (touches left), width=200 < 960 (half frame)
    """
    cfg = {**BASE_CFG, "reject_severe_edge_truncation": True}
    # Touch left edge (x1=0 ≤ margin=5), width=200 < 1920/2=960
    bbox = [0, 100, 200, 400]
    result = evaluate_crop_quality(bbox, FRAME_W, FRAME_H, cfg)
    assert result["accepted"] is False
    assert result["reason"] == "severely_truncated", f"Got: {result}"


# ---------------------------------------------------------------------------
# TEST 7 — Edge truncation disabled → same crop not rejected for truncation
# ---------------------------------------------------------------------------

def test_07_severe_truncation_disabled_not_rejected():
    """
    The same edge-touching crop from TEST 6 should NOT be rejected when
    reject_severe_edge_truncation=False — as long as it meets size thresholds.
    """
    cfg = {**BASE_CFG, "reject_severe_edge_truncation": False}
    # Same bbox as TEST 6 — touches left edge, width=200, height=300
    # width 200 >= min_w 40, height 300 >= min_h 100 → should pass size check
    bbox = [0, 100, 200, 400]
    result = evaluate_crop_quality(bbox, FRAME_W, FRAME_H, cfg)
    # With truncation disabled, the only remaining filter is size.
    # 200×300 passes size, so accepted.
    assert result["accepted"] is True, f"Expected accepted with truncation disabled, got: {result}"
    assert result["reason"] == "accepted"


# ---------------------------------------------------------------------------
# TEST 8 — Config values loaded correctly from config.yaml
# ---------------------------------------------------------------------------

def test_08_config_loaded_from_yaml():
    """load_quality_config() should return Phase 3.5 values from config.yaml."""
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if not config_path.exists():
        pytest.skip(f"config.yaml not found at {config_path}")

    cfg = load_quality_config(config_path)

    # Phase 3.5 values
    assert cfg["min_crop_width"]  == 40,  f"Expected 40, got {cfg['min_crop_width']}"
    assert cfg["min_crop_height"] == 100, f"Expected 100, got {cfg['min_crop_height']}"
    assert cfg["reject_severe_edge_truncation"] is True
    assert cfg["edge_margin_pixels"] == 5

    # All required keys present
    for key in ("min_crop_width", "min_crop_height",
                "reject_severe_edge_truncation", "edge_margin_pixels"):
        assert key in cfg, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# TEST 9 — Every rejected crop has an explicit non-empty reason
# ---------------------------------------------------------------------------

def test_09_rejected_crop_has_explicit_reason():
    """
    Any call to evaluate_crop_quality that returns accepted=False must
    include a non-empty reason string that is not 'accepted' and not vague.
    """
    cases = [
        None,                        # invalid_bbox
        [100, 100, 100, 300],        # degenerate bbox → invalid_bbox
        [200, 100, 239, 250],        # crop_too_small (width=39)
        [200, 100, 250, 199],        # crop_too_small (height=99)
        [0, 100, 200, 400],          # severely_truncated (left edge + small)
    ]
    vague_reasons = {"bad_crop", "rejected", "failed", "error", ""}

    for bbox in cases:
        result = evaluate_crop_quality(bbox, FRAME_W, FRAME_H, BASE_CFG)
        if not result["accepted"]:
            reason = result.get("reason", "")
            assert reason, f"reason is empty for bbox={bbox}"
            assert reason not in vague_reasons, (
                f"Vague reason '{reason}' for bbox={bbox}"
            )
            assert result["accepted"] is False
            assert "details" in result
            assert result["details"]  # non-empty details


# ---------------------------------------------------------------------------
# TEST 10 — Real C01 crops_metadata.json evaluates without crashing
# ---------------------------------------------------------------------------

def test_10_real_c01_metadata_no_crash():
    """
    Load the real crops_metadata.json (367 crops) and run evaluate_crop_quality
    on every record. No YOLO/ByteTrack involved — pure arithmetic.
    Asserts:
      - No exception is raised.
      - Every result dict has accepted, reason, details keys.
      - At least one crop is accepted (sanity check that filtering isn't broken).
    """
    metadata_path = Path(__file__).resolve().parents[2] / "dataset" / "crops_metadata.json"
    if not metadata_path.exists():
        pytest.skip(f"crops_metadata.json not found: {metadata_path}")

    with open(metadata_path) as f:
        records = json.load(f)

    assert records, "crops_metadata.json is empty"

    # Use a representative frame size for C01 (1920×1080)
    img_w, img_h = 1920, 1080
    cfg = load_quality_config(Path(__file__).resolve().parents[1] / "config.yaml")

    accepted_count = 0
    for rec in records:
        bbox   = rec["bbox"]
        result = evaluate_crop_quality(bbox, img_w, img_h, cfg)

        # All required keys must be present
        assert "accepted" in result
        assert "reason"   in result
        assert "details"  in result

        # reason must be one of the documented values
        assert result["reason"] in {
            "accepted", "invalid_bbox", "crop_too_small", "severely_truncated"
        }, f"Unexpected reason: {result['reason']}"

        if result["accepted"]:
            accepted_count += 1

    # We expect the majority of existing (valid) crops to be accepted
    assert accepted_count > 0, "All crops rejected — likely a config or logic error"
    print(
        f"\n[TEST 10] Real C01: {len(records)} records evaluated, "
        f"{accepted_count} accepted, {len(records) - accepted_count} rejected."
    )


# ---------------------------------------------------------------------------
# Helpers — lower-level function tests
# ---------------------------------------------------------------------------

class TestIsCropLargeEnough:
    def test_passes_when_both_above(self):
        assert is_crop_large_enough(50, 110, 40, 100) is True

    def test_passes_at_exact_minimum(self):
        assert is_crop_large_enough(40, 100, 40, 100) is True

    def test_fails_when_width_below(self):
        assert is_crop_large_enough(39, 110, 40, 100) is False

    def test_fails_when_height_below(self):
        assert is_crop_large_enough(50, 99, 40, 100) is False

    def test_fails_both_below(self):
        assert is_crop_large_enough(10, 10, 40, 100) is False


class TestIsSeverelyTruncated:
    """Tests for the four-edge rule."""

    def test_left_edge_small_crop_truncated(self):
        # x1=0, crop_w=100 < 1920/2=960 → truncated
        assert is_severely_truncated((0, 100, 100, 400), 1920, 1080, 5) is True

    def test_right_edge_small_crop_truncated(self):
        # x2=1920, crop_w=100 < 960 → truncated
        assert is_severely_truncated((1820, 100, 1920, 400), 1920, 1080, 5) is True

    def test_top_edge_small_crop_truncated(self):
        # y1=0, crop_h=200 < 1080/2=540 → truncated
        assert is_severely_truncated((100, 0, 300, 200), 1920, 1080, 5) is True

    def test_bottom_edge_small_crop_truncated(self):
        # y2=1080, crop_h=200 < 540 → truncated
        assert is_severely_truncated((100, 880, 300, 1080), 1920, 1080, 5) is True

    def test_left_edge_large_crop_not_truncated(self):
        # x1=0 BUT crop_w=1000 > 960 (majority of frame visible) → not truncated
        assert is_severely_truncated((0, 100, 1000, 900), 1920, 1080, 5) is False

    def test_far_from_edge_not_truncated(self):
        # Centred crop, no edge contact → not truncated
        assert is_severely_truncated((400, 100, 600, 800), 1920, 1080, 5) is False

    def test_within_margin_considered_edge(self):
        # x1=3 ≤ margin=5, crop_w=100 < 960 → truncated
        assert is_severely_truncated((3, 100, 103, 500), 1920, 1080, 5) is True

    def test_just_outside_margin_not_edge(self):
        # x1=6 > margin=5 → left edge not triggered
        assert is_severely_truncated((6, 100, 106, 500), 1920, 1080, 5) is False


# ---------------------------------------------------------------------------
# Allow running directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    sys.exit(result.returncode)
