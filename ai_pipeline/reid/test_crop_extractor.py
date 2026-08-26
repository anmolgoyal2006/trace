"""
test_crop_extractor.py — Phase 3.2
Tests for bounding-box clipping, minimum-size rejection, filename format,
metadata structure, and sampling integration.

Does NOT run YOLO or ByteTrack.

Run with:
    python ai_pipeline/reid/test_crop_extractor.py
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid.crop_extractor import clip_bbox, crop_name, extract_crops, load_reid_config
from reid.sampling import sample_track_frames

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pass_(label: str) -> None:
    print(f"  PASS  {label}")

def fail(label: str, msg: str = "") -> None:
    print(f"  FAIL  {label}" + (f": {msg}" if msg else ""))
    sys.exit(1)

def assert_eq(label, actual, expected):
    if actual != expected:
        fail(label, f"expected {expected!r}, got {actual!r}")
    pass_(label)

def assert_true(label, cond, msg=""):
    if not cond:
        fail(label, msg)
    pass_(label)

def assert_none(label, val):
    if val is not None:
        fail(label, f"expected None, got {val!r}")
    pass_(label)

def make_record(track_id, frame, bbox, camera_id="C01"):
    return {
        "camera_id": camera_id,
        "frame": frame,
        "timestamp": "10:02:00.00",
        "track_id": track_id,
        "bbox": bbox,
        "confidence": 0.90,
    }

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clip_bbox_inside():
    print("\n[TEST 1] bbox fully inside frame")
    result = clip_bbox([100.0, 200.0, 400.0, 600.0], 1920, 1080)
    assert_eq("x1", result[0], 100)
    assert_eq("y1", result[1], 200)
    assert_eq("x2", result[2], 400)
    assert_eq("y2", result[3], 600)


def test_clip_bbox_partial_outside():
    print("\n[TEST 2] bbox partially outside frame — should be clipped")
    result = clip_bbox([-10.0, 200.0, 500.0, 1100.0], 1920, 1080)
    assert_eq("x1 clipped to 0",    result[0], 0)
    assert_eq("y1 unchanged",        result[1], 200)
    assert_eq("x2 unchanged",        result[2], 500)
    assert_eq("y2 clipped to 1080",  result[3], 1080)


def test_clip_bbox_fully_outside():
    print("\n[TEST 3] bbox completely outside frame — should return None")
    result = clip_bbox([2000.0, 0.0, 2500.0, 500.0], 1920, 1080)
    assert_none("fully outside → None", result)


def test_clip_bbox_degenerate():
    print("\n[TEST 4] degenerate bbox (x2 <= x1) — should return None")
    result = clip_bbox([400.0, 100.0, 200.0, 300.0], 1920, 1080)
    assert_none("degenerate → None", result)


def test_min_crop_rejected():
    print("\n[TEST 5] crop too small — should be skipped")
    # Create a 1920x1080 black synthetic frame
    img      = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img_h, img_w = img.shape[:2]

    # A valid but tiny bbox: 20×40 px (below min 30×60)
    bbox = [100.0, 100.0, 120.0, 140.0]
    clipped = clip_bbox(bbox, img_w, img_h)
    assert_true("clipped is not None", clipped is not None)
    x1, y1, x2, y2 = clipped
    w, h = x2 - x1, y2 - y1
    assert_true("width < 30",  w < 30,  f"w={w}")
    assert_true("height < 60", h < 60,  f"h={h}")
    print("  INFO  Crop would be skipped: too small", w, "x", h)


def test_valid_crop_saved():
    print("\n[TEST 6] valid crop is saved correctly")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Synthetic 1920x1080 coloured frame
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        video_path = tmp / "fake.mp4"
        out_path   = tmp / "crop.jpg"

        # Write a 3-frame synthetic video
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25, (1920, 1080),
        )
        for _ in range(3):
            writer.write(img)
        writer.release()

        # Extract crop from frame 1, bbox covers a 200×400 region
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        ret, frame = cap.read()
        cap.release()
        assert_true("frame read ok", ret)

        x1, y1, x2, y2 = 100, 100, 300, 500
        crop = frame[y1:y2, x1:x2]
        ok = cv2.imwrite(str(out_path), crop)
        assert_true("imwrite succeeded", ok)

        saved = cv2.imread(str(out_path))
        assert_true("saved file readable", saved is not None)
        assert_eq("crop height", saved.shape[0], 400)
        assert_eq("crop width",  saved.shape[1], 200)


def test_crop_filename_format():
    print("\n[TEST 7] filename format is correct")
    assert_eq("C01 track1 frame142",
              crop_name("C01", 1, 142),
              "C01_track1_frame0142.jpg")
    assert_eq("C02 track37 frame0",
              crop_name("C02", 37, 0),
              "C02_track37_frame0000.jpg")
    assert_eq("C03 track110 frame632",
              crop_name("C03", 110, 632),
              "C03_track110_frame0632.jpg")


def test_metadata_fields():
    print("\n[TEST 8] metadata record has all required fields")
    required = {"camera_id", "track_id", "frame", "timestamp",
                "bbox", "confidence", "crop_path"}

    # Build a mock metadata entry
    entry = {
        "camera_id":  "C01",
        "track_id":   1,
        "frame":      142,
        "timestamp":  "10:02:04.89",
        "bbox":       [312.4, 88.1, 489.7, 412.3],
        "confidence": 0.94,
        "crop_path":  "dataset/crops/C01_track1_frame0142.jpg",
    }
    missing = required - entry.keys()
    assert_true("all fields present", not missing, f"missing: {missing}")
    assert_true("crop_path is str",   isinstance(entry["crop_path"], str))
    assert_true("bbox is list",       isinstance(entry["bbox"], list))
    assert_eq("bbox length",          len(entry["bbox"]), 4)


def test_sampling_used():
    print("\n[TEST 9] Phase 3.1 sampling is actually invoked")
    records = (
        [make_record(1, f, [100.0, 100.0, 300.0, 500.0]) for f in range(100)]
        + [make_record(2, f, [500.0, 100.0, 700.0, 500.0]) for f in range(50)]
    )
    # K=5: each track should yield at most 5
    sampled = sample_track_frames(records, 5)

    by_track = {}
    for r in sampled:
        by_track.setdefault(r["track_id"], []).append(r["frame"])

    assert_eq("track 1 count", len(by_track[1]), 5)
    assert_eq("track 2 count", len(by_track[2]), 5)
    assert_eq("track 1 first", by_track[1][0],   0)
    assert_eq("track 1 last",  by_track[1][-1],  99)
    assert_eq("track 2 first", by_track[2][0],   0)
    assert_eq("track 2 last",  by_track[2][-1],  49)


def test_full_pipeline_synthetic():
    """End-to-end test on a synthetic video — no YOLO, no ByteTrack."""
    print("\n[TEST 10] Full pipeline — synthetic video + mock tracks")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Write 20-frame synthetic video (1920x1080)
        video_path = tmp / "synth.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25, (1920, 1080),
        )
        for i in range(20):
            frame = np.full((1080, 1920, 3), i * 10, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # Two tracks, each with 10 frames; K=3 → 3 crops each = 6 total
        tracks = (
            [make_record(1, f, [100.0, 100.0, 400.0, 700.0]) for f in range(10)]
            + [make_record(2, f, [800.0, 100.0, 1200.0, 700.0]) for f in range(10, 20)]
        )
        tracks_path  = tmp / "tracks.json"
        output_dir   = tmp / "crops"
        metadata_path = tmp / "meta.json"

        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        reid_cfg = {"crops_per_track": 3, "min_crop_width": 30, "min_crop_height": 60}
        extract_crops(video_path, tracks_path, output_dir, metadata_path, reid_cfg)

        # Verify
        assert_true("output dir exists", output_dir.exists())
        crops = list(output_dir.glob("*.jpg"))
        assert_true("6 crops saved", len(crops) == 6, f"got {len(crops)}")

        with open(metadata_path) as f:
            meta = json.load(f)
        assert_eq("metadata records", len(meta), 6)
        required_fields = {"camera_id","track_id","frame","timestamp",
                           "bbox","confidence","crop_path"}
        for entry in meta:
            missing = required_fields - entry.keys()
            assert_true("all fields in metadata entry", not missing, str(missing))

        # Each crop should be readable and non-empty
        for entry in meta:
            img = cv2.imread(entry["crop_path"])
            assert_true(f"crop readable: {Path(entry['crop_path']).name}", img is not None)

        print(f"  INFO  Crops: {[c.name for c in sorted(crops)]}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_clip_bbox_inside()
    test_clip_bbox_partial_outside()
    test_clip_bbox_fully_outside()
    test_clip_bbox_degenerate()
    test_min_crop_rejected()
    test_valid_crop_saved()
    test_crop_filename_format()
    test_metadata_fields()
    test_sampling_used()
    test_full_pipeline_synthetic()

    print("\n=== All tests passed. ===")
