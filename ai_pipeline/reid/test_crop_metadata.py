"""
test_crop_metadata.py — Phase 3.3
Tests for validate_crop_metadata.py logic.

Does NOT modify real crop images or crops_metadata.json.
Uses temporary directories for synthetic cases.

Run with:
    python ai_pipeline/reid/test_crop_metadata.py
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Allow importing from sibling packages
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid.validate_crop_metadata import validate, validate_record, REQUIRED_FIELDS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]   # Trace/


def _pass(label: str) -> None:
    print(f"  PASS  {label}")


def _fail(label: str, msg: str = "") -> None:
    print(f"  FAIL  {label}" + (f": {msg}" if msg else ""))
    sys.exit(1)


def assert_true(label: str, cond: bool, msg: str = "") -> None:
    if not cond:
        _fail(label, msg)
    _pass(label)


def assert_eq(label: str, actual, expected) -> None:
    if actual != expected:
        _fail(label, f"expected {expected!r}, got {actual!r}")
    _pass(label)


def make_jpg(path: Path, w: int = 100, h: int = 200) -> None:
    """Write a small synthetic JPEG to path."""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def good_record(crop_path: str) -> dict:
    """Return a fully valid metadata record."""
    return {
        "crop_path":            crop_path,
        "camera_id":            "C01",
        "track_id":             1,
        "frame":                142,
        "timestamp":            "10:02:04.89",
        "bbox":                 [312.4, 88.1, 489.7, 412.3],
        "detection_confidence": 0.94,
    }


# ---------------------------------------------------------------------------
# TEST 1 — valid record passes
# ---------------------------------------------------------------------------

def test_valid_record_passes():
    print("\n[TEST 1] Valid record passes all checks")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()
        make_jpg(crops_dir / "C01_track1_frame0142.jpg")

        rec = good_record(f"{crops_dir}/C01_track1_frame0142.jpg")
        # For validate_record, crop_path just needs to resolve inside crops_dir
        # Use a relative-style path from a fake repo root = tmp
        rec["crop_path"] = "crops/C01_track1_frame0142.jpg"
        errs = validate_record(rec, 0, crops_dir, Path(tmp))
        assert_eq("no errors", errs, [])


# ---------------------------------------------------------------------------
# TEST 2 — missing required field fails
# ---------------------------------------------------------------------------

def test_missing_field_fails():
    print("\n[TEST 2] Missing required field is detected")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()
        make_jpg(crops_dir / "C01_track1_frame0142.jpg")

        for field in REQUIRED_FIELDS:
            rec = good_record("crops/C01_track1_frame0142.jpg")
            del rec[field]
            errs = validate_record(rec, 0, crops_dir, Path(tmp))
            assert_true(
                f"missing '{field}' caught",
                any(field in e for e in errs),
                f"errors={errs}",
            )


# ---------------------------------------------------------------------------
# TEST 3 — invalid bbox length fails
# ---------------------------------------------------------------------------

def test_invalid_bbox_length():
    print("\n[TEST 3] Invalid bbox length is detected")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()
        make_jpg(crops_dir / "C01_track1_frame0142.jpg")

        rec = good_record("crops/C01_track1_frame0142.jpg")
        rec["bbox"] = [1.0, 2.0, 3.0]   # only 3 elements
        errs = validate_record(rec, 0, crops_dir, Path(tmp))
        assert_true("bbox length error caught", any("bbox" in e for e in errs), str(errs))


# ---------------------------------------------------------------------------
# TEST 4 — confidence outside [0, 1] fails
# ---------------------------------------------------------------------------

def test_confidence_out_of_range():
    print("\n[TEST 4] detection_confidence outside [0,1] is detected")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()
        make_jpg(crops_dir / "C01_track1_frame0142.jpg")

        for bad_conf in [-0.1, 1.01, 2.0, -100]:
            rec = good_record("crops/C01_track1_frame0142.jpg")
            rec["detection_confidence"] = bad_conf
            errs = validate_record(rec, 0, crops_dir, Path(tmp))
            assert_true(
                f"conf={bad_conf} caught",
                any("detection_confidence" in e for e in errs),
                str(errs),
            )


# ---------------------------------------------------------------------------
# TEST 5 — missing crop file is detected
# ---------------------------------------------------------------------------

def test_missing_crop_file():
    print("\n[TEST 5] Missing crop file is detected by full validator")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()
        # No actual JPEG created

        meta_path = Path(tmp) / "meta.json"
        records = [good_record("crops/C01_track1_frame0142.jpg")]
        with open(meta_path, "w") as f:
            json.dump(records, f)

        result = validate(meta_path, crops_dir, Path(tmp))
        assert_true("validation fails on missing file", not result)


# ---------------------------------------------------------------------------
# TEST 6 — orphan image is detected
# ---------------------------------------------------------------------------

def test_orphan_image_detected():
    print("\n[TEST 6] Orphan image (no metadata entry) is detected")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()

        # Two images but only one in metadata
        make_jpg(crops_dir / "C01_track1_frame0000.jpg")
        make_jpg(crops_dir / "C01_track2_frame0100.jpg")   # orphan

        meta_path = Path(tmp) / "meta.json"
        records = [good_record("crops/C01_track1_frame0000.jpg")]
        with open(meta_path, "w") as f:
            json.dump(records, f)

        result = validate(meta_path, crops_dir, Path(tmp))
        assert_true("validation fails on orphan image", not result)


# ---------------------------------------------------------------------------
# TEST 7 — duplicate crop_path is detected
# ---------------------------------------------------------------------------

def test_duplicate_crop_path():
    print("\n[TEST 7] Duplicate crop_path values are detected")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()
        make_jpg(crops_dir / "C01_track1_frame0000.jpg")

        meta_path = Path(tmp) / "meta.json"
        records = [
            good_record("crops/C01_track1_frame0000.jpg"),
            good_record("crops/C01_track1_frame0000.jpg"),   # duplicate
        ]
        with open(meta_path, "w") as f:
            json.dump(records, f)

        result = validate(meta_path, crops_dir, Path(tmp))
        assert_true("validation fails on duplicate path", not result)


# ---------------------------------------------------------------------------
# TEST 8 — unreadable image is detected
# ---------------------------------------------------------------------------

def test_unreadable_image():
    print("\n[TEST 8] Unreadable image (corrupt file) is detected")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()

        bad_jpg = crops_dir / "C01_track1_frame0000.jpg"
        bad_jpg.write_bytes(b"not a real jpeg")  # corrupt file

        meta_path = Path(tmp) / "meta.json"
        records = [good_record("crops/C01_track1_frame0000.jpg")]
        with open(meta_path, "w") as f:
            json.dump(records, f)

        result = validate(meta_path, crops_dir, Path(tmp))
        assert_true("validation fails on unreadable image", not result)


# ---------------------------------------------------------------------------
# TEST 9 — absolute / out-of-dir path is rejected
# ---------------------------------------------------------------------------

def test_path_outside_crops_dir_rejected():
    print("\n[TEST 9] Path resolving outside crops_dir is rejected")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()

        # Absolute path
        rec_abs = good_record("C:/some/absolute/path/file.jpg")
        errs_abs = validate_record(rec_abs, 0, crops_dir, Path(tmp))
        assert_true("absolute path caught", any("absolute" in e for e in errs_abs), str(errs_abs))

        # Path traversal that escapes crops_dir
        rec_trav = good_record("../outside/file.jpg")
        errs_trav = validate_record(rec_trav, 0, crops_dir, Path(tmp))
        assert_true("path traversal caught",
                    any("outside" in e or "crops_dir" in e for e in errs_trav),
                    str(errs_trav))


# ---------------------------------------------------------------------------
# TEST 10 — real C01 data passes
# ---------------------------------------------------------------------------

def test_real_c01_data():
    print("\n[TEST 10] Real C01 crops and metadata pass validation")

    meta_path = REPO_ROOT / "dataset" / "crops_metadata.json"
    crops_dir = REPO_ROOT / "dataset" / "crops"

    if not meta_path.exists():
        print("  SKIP  crops_metadata.json not found — run crop_extractor.py first")
        return
    if not crops_dir.exists():
        print("  SKIP  dataset/crops/ not found — run crop_extractor.py first")
        return

    result = validate(meta_path, crops_dir, REPO_ROOT)
    assert_true("real C01 validation passes", result)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_valid_record_passes()
    test_missing_field_fails()
    test_invalid_bbox_length()
    test_confidence_out_of_range()
    test_missing_crop_file()
    test_orphan_image_detected()
    test_duplicate_crop_path()
    test_unreadable_image()
    test_path_outside_crops_dir_rejected()
    test_real_c01_data()

    print("\n=== All tests passed. ===")
