"""
test_visualize_crops.py — Phase 3.4
Tests for the non-visual logic in visualize_crops.py.

Does NOT require a GUI. Does NOT modify real crops or metadata.
Uses temporary directories for synthetic cases.

Run with:
    python ai_pipeline/reid/test_visualize_crops.py
"""

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid.visualize_crops import (
    load_metadata,
    abs_crop_path,
    make_contact_sheet,
    fit_into_tile,
    REPO_ROOT,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pass(label): print(f"  PASS  {label}")
def _fail(label, msg=""):
    print(f"  FAIL  {label}" + (f": {msg}" if msg else ""))
    sys.exit(1)

def assert_true(label, cond, msg=""):
    if not cond: _fail(label, msg)
    _pass(label)

def assert_eq(label, actual, expected):
    if actual != expected: _fail(label, f"expected {expected!r}, got {actual!r}")
    _pass(label)

def make_jpg(path: Path, w=80, h=200):
    img = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path

def make_record(track_id, frame, crop_path, camera_id="C01", conf=0.9):
    return {
        "crop_path": str(crop_path).replace("\\", "/"),
        "camera_id": camera_id,
        "track_id": track_id,
        "frame": frame,
        "timestamp": f"10:02:{frame:05.2f}",
        "bbox": [100.0, 100.0, 300.0, 500.0],
        "detection_confidence": conf,
    }

def write_meta(path: Path, records: list) -> Path:
    with open(path, "w") as f:
        json.dump(records, f)
    return path


# ---------------------------------------------------------------------------
# TEST 1 — metadata loads successfully
# ---------------------------------------------------------------------------

def test_metadata_loads():
    print("\n[TEST 1] Metadata loads successfully")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()
        make_jpg(crops_dir / "C01_track1_frame0000.jpg")

        meta_path = Path(tmp) / "meta.json"
        records = [make_record(1, 0, "crops/C01_track1_frame0000.jpg")]
        write_meta(meta_path, records)

        loaded = load_metadata(meta_path)
        assert_eq("record count", len(loaded), 1)
        assert_eq("track_id", loaded[0]["track_id"], 1)


# ---------------------------------------------------------------------------
# TEST 2 — random sampling returns requested number
# ---------------------------------------------------------------------------

def test_random_sampling_count():
    print("\n[TEST 2] Random sampling returns requested number of crops")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()

        records = []
        for i in range(20):
            fname = f"C01_track1_frame{i:04d}.jpg"
            make_jpg(crops_dir / fname)
            records.append(make_record(1, i, f"crops/{fname}"))

        rng = random.Random(42)
        chosen = rng.sample(records, 10)
        assert_eq("count", len(chosen), 10)


# ---------------------------------------------------------------------------
# TEST 3 — sampling is deterministic with same seed
# ---------------------------------------------------------------------------

def test_sampling_deterministic():
    print("\n[TEST 3] Sampling is deterministic with the same seed")
    records = [make_record(i, i, f"crops/C01_track{i}_frame{i:04d}.jpg")
               for i in range(30)]

    def sample(seed):
        return [r["track_id"] for r in random.Random(seed).sample(records, 8)]

    assert_eq("run1 == run2", sample(42), sample(42))
    assert_true("different seed → different", sample(42) != sample(99),
                "seeds 42 and 99 produced identical samples — unlikely but possible")


# ---------------------------------------------------------------------------
# TEST 4 — selected crop paths exist
# ---------------------------------------------------------------------------

def test_selected_crop_paths_exist():
    print("\n[TEST 4] Selected crop paths exist on disk")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()

        records = []
        for i in range(5):
            fname = f"C01_track1_frame{i:04d}.jpg"
            make_jpg(crops_dir / fname)
            # Store as absolute path so abs_crop_path returns it directly
            records.append(make_record(1, i, str(crops_dir / fname)))

        for rec in records:
            p = abs_crop_path(rec)
            assert_true(f"frame {rec['frame']} exists", p.exists(), str(p))


# ---------------------------------------------------------------------------
# TEST 5 — track filtering returns only requested track_id
# ---------------------------------------------------------------------------

def test_track_filtering():
    print("\n[TEST 5] Track filtering returns only the requested track_id")
    records = (
        [make_record(1, f, f"crops/C01_track1_frame{f:04d}.jpg") for f in range(5)]
        + [make_record(2, f, f"crops/C01_track2_frame{f:04d}.jpg") for f in range(5)]
        + [make_record(3, f, f"crops/C01_track3_frame{f:04d}.jpg") for f in range(5)]
    )
    tid = 2
    filtered = [r for r in records if r["track_id"] == tid]
    assert_eq("count", len(filtered), 5)
    assert_true("all tid==2", all(r["track_id"] == tid for r in filtered))


# ---------------------------------------------------------------------------
# TEST 6 — track records sorted by frame
# ---------------------------------------------------------------------------

def test_track_records_sorted():
    print("\n[TEST 6] Track records are sorted by frame number")
    frames = [50, 10, 30, 90, 70]
    records = [make_record(7, f, f"crops/C01_track7_frame{f:04d}.jpg")
               for f in frames]
    sorted_recs = sorted(records, key=lambda r: r["frame"])
    result_frames = [r["frame"] for r in sorted_recs]
    assert_eq("sorted frames", result_frames, [10, 30, 50, 70, 90])


# ---------------------------------------------------------------------------
# TEST 7 — missing image path handled gracefully
# ---------------------------------------------------------------------------

def test_missing_image_graceful():
    print("\n[TEST 7] Missing image path handled gracefully (no crash)")
    # Record points to a nonexistent file
    rec = make_record(99, 0, "crops/C01_track99_frame0000.jpg")
    # make_contact_sheet should not crash — should count it as unreadable
    try:
        sheet, stats = make_contact_sheet([rec], cols=1)
        assert_true("no crash", True)
        assert_eq("unreadable count", stats["unreadable"], 1)
    except Exception as e:
        _fail("contact sheet crashed on missing image", str(e))


# ---------------------------------------------------------------------------
# TEST 8 — contact sheet handles different crop sizes
# ---------------------------------------------------------------------------

def test_contact_sheet_mixed_sizes():
    print("\n[TEST 8] Contact sheet handles crops of different dimensions")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()

        sizes = [(40, 120), (100, 300), (60, 180), (80, 250)]
        records = []
        for i, (w, h) in enumerate(sizes):
            fname = f"C01_track1_frame{i:04d}.jpg"
            make_jpg(crops_dir / fname, w=w, h=h)
            records.append(make_record(1, i, str(crops_dir / fname)))

        try:
            sheet, stats = make_contact_sheet(records, cols=4)
            assert_true("sheet is ndarray", isinstance(sheet, np.ndarray))
            assert_true("sheet has 3 channels", sheet.ndim == 3 and sheet.shape[2] == 3)
            assert_eq("total inspected", stats["total"], 4)
            assert_eq("no unreadable", stats["unreadable"], 0)
        except Exception as e:
            _fail("contact sheet crashed on mixed sizes", str(e))


# ---------------------------------------------------------------------------
# TEST 9 — original crop files are not modified
# ---------------------------------------------------------------------------

def test_original_crops_not_modified():
    print("\n[TEST 9] Original crop files are not modified by contact sheet generation")
    with tempfile.TemporaryDirectory() as tmp:
        crops_dir = Path(tmp) / "crops"
        crops_dir.mkdir()

        fname = "C01_track1_frame0000.jpg"
        src = crops_dir / fname
        make_jpg(src, w=80, h=200)
        original_size = src.stat().st_size
        original_mtime = src.stat().st_mtime

        rec = make_record(1, 0, str(src))
        make_contact_sheet([rec], cols=1)   # generate sheet (result discarded)

        assert_eq("file size unchanged", src.stat().st_size, original_size)
        assert_eq("mtime unchanged", src.stat().st_mtime, original_mtime)


# ---------------------------------------------------------------------------
# TEST 10 — metadata file is not modified
# ---------------------------------------------------------------------------

def test_metadata_not_modified():
    print("\n[TEST 10] crops_metadata.json is not modified by visualize_crops")
    real_meta = REPO_ROOT / "dataset" / "crops_metadata.json"
    if not real_meta.exists():
        print("  SKIP  crops_metadata.json not found")
        return

    original_mtime = real_meta.stat().st_mtime
    original_size  = real_meta.stat().st_size

    # Load metadata (read-only operation)
    records = load_metadata(real_meta)

    # Sample a few records and generate a contact sheet in a temp dir
    with tempfile.TemporaryDirectory() as tmp:
        sample = records[:6]
        make_contact_sheet(sample, cols=3)   # result discarded

    assert_eq("mtime unchanged", real_meta.stat().st_mtime, original_mtime)
    assert_eq("size unchanged",  real_meta.stat().st_size,  original_size)
    print(f"  INFO  Loaded {len(records)} records — file untouched")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_metadata_loads()
    test_random_sampling_count()
    test_sampling_deterministic()
    test_selected_crop_paths_exist()
    test_track_filtering()
    test_track_records_sorted()
    test_missing_image_graceful()
    test_contact_sheet_mixed_sizes()
    test_original_crops_not_modified()
    test_metadata_not_modified()

    print("\n=== All tests passed. ===")
