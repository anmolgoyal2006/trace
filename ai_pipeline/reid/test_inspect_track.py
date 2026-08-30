"""
Phase 5.4.1 — Pytest Tests for inspect_track.py
================================================
All tests use synthetic data or the real Phase 3 crops (read-only).
No production data is modified.  No GPU required.

Run from the workspace root:
    pytest ai_pipeline/reid/test_inspect_track.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

# Ensure repo root is on sys.path (matches pattern in test_track_aggregation.py)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ai_pipeline.reid.inspect_track import (
    CropRecord,
    LoadResult,
    find_crops_for_track,
    load_crop,
    make_contact_sheet,
    inspect_tracks,
    _scale_to_height,
    _make_crop_tile,
    _make_error_tile,
    TILE_HEIGHT,
    LABEL_HEIGHT,
)

# ---------------------------------------------------------------------------
# Paths to real Phase 3 data (read-only)
# ---------------------------------------------------------------------------

REAL_META   = _REPO_ROOT / "dataset" / "crops_metadata_phase3_final.json"
REAL_CROPS  = _REPO_ROOT / "dataset" / "crops_phase3_final"

_REAL_DATA_AVAILABLE = REAL_META.exists() and REAL_CROPS.exists()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    return tmp_path / "inspection"


@pytest.fixture
def synthetic_meta(tmp_path) -> tuple[list[dict], Path, Path]:
    """
    Create a small synthetic metadata file and matching JPEG crops.
    Returns (metadata_list, meta_path, crops_dir).
    """
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()

    records = []
    for i, (frame, conf) in enumerate([(10, 0.90), (20, 0.85), (30, 0.70)]):
        fname = f"C01_track99_frame{frame:04d}.jpg"
        fpath = crops_dir / fname

        # Create a small synthetic RGB image
        img = Image.new("RGB", (80 + i * 20, 200 + i * 10), color=(i * 60, 100, 200))
        img.save(str(fpath), format="JPEG", quality=90)

        records.append({
            "crop_path": f"crops/{fname}",
            "camera_id": "C01",
            "track_id": 99,
            "frame": frame,
            "timestamp": f"10:00:{frame:02d}.00",
            "bbox": [0.0, 0.0, float(80 + i * 20), float(200 + i * 10)],
            "detection_confidence": conf,
        })

    # Add a record for a different track (should be ignored)
    records.append({
        "crop_path": "crops/other.jpg",
        "camera_id": "C01",
        "track_id": 50,
        "frame": 5,
        "timestamp": "10:00:00.00",
        "bbox": [0.0, 0.0, 50.0, 150.0],
        "detection_confidence": 0.80,
    })

    meta_path = tmp_path / "meta.json"
    with open(meta_path, "w") as fh:
        json.dump(records, fh)

    return records, meta_path, crops_dir


# ---------------------------------------------------------------------------
# 1. find_crops_for_track — basic location by track_id
# ---------------------------------------------------------------------------

class TestFindCropsByTrackId:
    def test_returns_correct_number_of_crops(self, synthetic_meta, tmp_path):
        _, meta_path, crops_dir = synthetic_meta
        with open(meta_path) as fh:
            meta = json.load(fh)
        result = find_crops_for_track(meta, 99, tmp_path)
        assert len(result) == 3

    def test_ignores_other_tracks(self, synthetic_meta, tmp_path):
        _, meta_path, crops_dir = synthetic_meta
        with open(meta_path) as fh:
            meta = json.load(fh)
        result = find_crops_for_track(meta, 50, tmp_path)
        assert len(result) == 1

    def test_returns_empty_for_nonexistent_track(self, synthetic_meta, tmp_path):
        _, meta_path, _ = synthetic_meta
        with open(meta_path) as fh:
            meta = json.load(fh)
        result = find_crops_for_track(meta, 9999, tmp_path)
        assert result == []

    def test_sorted_by_frame_ascending(self, synthetic_meta, tmp_path):
        _, meta_path, _ = synthetic_meta
        with open(meta_path) as fh:
            meta = json.load(fh)
        result = find_crops_for_track(meta, 99, tmp_path)
        frames = [r.frame for r in result]
        assert frames == sorted(frames)

    def test_all_fields_populated(self, synthetic_meta, tmp_path):
        _, meta_path, _ = synthetic_meta
        with open(meta_path) as fh:
            meta = json.load(fh)
        result = find_crops_for_track(meta, 99, tmp_path)
        for rec in result:
            assert rec.track_id == 99
            assert isinstance(rec.frame, int)
            assert isinstance(rec.timestamp, str)
            assert isinstance(rec.detection_confidence, float)
            assert isinstance(rec.full_path, Path)


# ---------------------------------------------------------------------------
# 2. load_crop — handles missing and corrupt crops gracefully
# ---------------------------------------------------------------------------

class TestLoadCrop:
    def test_loads_readable_image(self, synthetic_meta, tmp_path):
        _, _, crops_dir = synthetic_meta
        rec = CropRecord(
            crop_path="crops/C01_track99_frame0010.jpg",
            full_path=crops_dir / "C01_track99_frame0010.jpg",
            track_id=99, frame=10, timestamp="10:00:10.00",
            detection_confidence=0.90, bbox=[0, 0, 80, 200],
        )
        result = load_crop(rec)
        assert result.error is None
        assert result.image is not None
        assert result.image.mode == "RGB"

    def test_returns_none_image_for_missing_file(self, tmp_path):
        rec = CropRecord(
            crop_path="missing.jpg",
            full_path=tmp_path / "missing.jpg",
            track_id=1, frame=0, timestamp="", detection_confidence=0.0, bbox=[],
        )
        result = load_crop(rec)
        assert result.image is None
        assert result.error is not None
        assert "not found" in result.error.lower() or "missing" in result.error.lower() or len(result.error) > 0

    def test_returns_none_image_for_corrupt_file(self, tmp_path):
        corrupt = tmp_path / "corrupt.jpg"
        corrupt.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 10)   # truncated JPEG
        rec = CropRecord(
            crop_path="corrupt.jpg",
            full_path=corrupt,
            track_id=1, frame=0, timestamp="", detection_confidence=0.0, bbox=[],
        )
        result = load_crop(rec)
        assert result.image is None
        assert result.error is not None

    def test_does_not_raise_on_any_failure(self, tmp_path):
        """load_crop must never propagate an exception."""
        rec = CropRecord(
            crop_path="nonexistent.jpg",
            full_path=tmp_path / "nonexistent.jpg",
            track_id=1, frame=0, timestamp="", detection_confidence=0.0, bbox=[],
        )
        try:
            result = load_crop(rec)
        except Exception as exc:
            pytest.fail(f"load_crop raised an exception: {exc}")

    def test_result_has_record_field(self, synthetic_meta, tmp_path):
        _, _, crops_dir = synthetic_meta
        rec = CropRecord(
            crop_path="crops/C01_track99_frame0010.jpg",
            full_path=crops_dir / "C01_track99_frame0010.jpg",
            track_id=99, frame=10, timestamp="10:00:10.00",
            detection_confidence=0.90, bbox=[0, 0, 80, 200],
        )
        result = load_crop(rec)
        assert result.record is rec


# ---------------------------------------------------------------------------
# 3. Contact sheet generation
# ---------------------------------------------------------------------------

class TestMakeContactSheet:
    def _make_load_results(self, n: int) -> list[LoadResult]:
        """Produce n LoadResults with simple synthetic images."""
        results = []
        for i in range(n):
            img = Image.new("RGB", (60 + i * 15, 180 + i * 5), (i * 40, 80, 180))
            rec = CropRecord(
                crop_path=f"crop_{i}.jpg",
                full_path=Path(f"crop_{i}.jpg"),
                track_id=1, frame=i * 10, timestamp=f"10:00:{i:02d}.00",
                detection_confidence=0.80 - i * 0.05, bbox=[0, 0, 60, 180],
            )
            results.append(LoadResult(record=rec, image=img, error=None))
        return results

    def test_output_file_created(self, tmp_output):
        results = self._make_load_results(3)
        out = tmp_output / "track_1_contact_sheet.jpg"
        saved = make_contact_sheet(1, results, out)
        assert saved.exists()

    def test_output_is_readable_jpeg(self, tmp_output):
        results = self._make_load_results(3)
        out = tmp_output / "track_1_contact_sheet.jpg"
        make_contact_sheet(1, results, out)
        img = Image.open(out)
        assert img.format == "JPEG"
        assert img.mode == "RGB"

    def test_output_not_empty(self, tmp_output):
        results = self._make_load_results(3)
        out = tmp_output / "track_1_contact_sheet.jpg"
        make_contact_sheet(1, results, out)
        assert out.stat().st_size > 1000

    def test_single_crop_contact_sheet(self, tmp_output):
        """Contact sheet works with exactly one crop."""
        results = self._make_load_results(1)
        out = tmp_output / "track_1_single.jpg"
        saved = make_contact_sheet(1, results, out)
        assert saved.exists()

    def test_five_crop_contact_sheet(self, tmp_output):
        """Contact sheet works with five crops (typical track size)."""
        results = self._make_load_results(5)
        out = tmp_output / "track_1_five.jpg"
        saved = make_contact_sheet(1, results, out)
        assert saved.exists()

    def test_mixed_readable_unreadable(self, tmp_output):
        """Contact sheet handles a mix of readable and unreadable crops."""
        readable = self._make_load_results(2)
        rec_bad = CropRecord(
            crop_path="bad.jpg", full_path=Path("bad.jpg"),
            track_id=1, frame=99, timestamp="", detection_confidence=0.0, bbox=[],
        )
        unreadable = LoadResult(record=rec_bad, image=None, error="File not found")
        results = readable + [unreadable]
        out = tmp_output / "track_1_mixed.jpg"
        saved = make_contact_sheet(1, results, out)
        assert saved.exists()

    def test_output_dir_created_if_missing(self, tmp_path):
        """make_contact_sheet creates the output directory if it does not exist."""
        results = self._make_load_results(2)
        nested = tmp_path / "a" / "b" / "c"
        out = nested / "sheet.jpg"
        make_contact_sheet(1, results, out)
        assert out.exists()

    def test_does_not_modify_source_images(self, tmp_output):
        """The original crop images must be pixel-identical before and after."""
        img_orig = Image.new("RGB", (80, 200), (100, 150, 200))
        px_before = list(img_orig.getdata())
        rec = CropRecord(
            crop_path="crop.jpg", full_path=Path("crop.jpg"),
            track_id=1, frame=0, timestamp="", detection_confidence=0.9, bbox=[],
        )
        lr = LoadResult(record=rec, image=img_orig, error=None)
        make_contact_sheet(1, [lr], tmp_output / "sheet.jpg")
        px_after = list(img_orig.getdata())
        assert px_before == px_after


# ---------------------------------------------------------------------------
# 4. _scale_to_height helper
# ---------------------------------------------------------------------------

class TestScaleToHeight:
    def test_output_height_matches_target(self):
        img = Image.new("RGB", (100, 200))
        scaled = _scale_to_height(img, 300, 400)
        assert scaled.height == 300

    def test_width_capped_at_max(self):
        img = Image.new("RGB", (1000, 100))   # very wide
        scaled = _scale_to_height(img, 300, 150)
        assert scaled.width <= 150

    def test_proportional_scaling_below_max(self):
        img = Image.new("RGB", (100, 200))    # 1:2 aspect ratio
        scaled = _scale_to_height(img, 200, 500)
        # expected width = 100 * (200/200) = 100
        assert scaled.height == 200
        assert scaled.width == 100


# ---------------------------------------------------------------------------
# 5. Tracks with different numbers of crops
# ---------------------------------------------------------------------------

class TestVariableCropCounts:
    def _run_for_n_crops(self, n: int, tmp_output: Path) -> Path:
        img = Image.new("RGB", (80, 200), (50, 100, 150))
        results = []
        for i in range(n):
            rec = CropRecord(
                crop_path=f"c{i}.jpg", full_path=Path(f"c{i}.jpg"),
                track_id=7, frame=i, timestamp="",
                detection_confidence=0.8, bbox=[],
            )
            results.append(LoadResult(record=rec, image=img, error=None))
        out = tmp_output / f"track_7_{n}_crops.jpg"
        return make_contact_sheet(7, results, out)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
    def test_contact_sheet_for_n_crops(self, n, tmp_output):
        path = self._run_for_n_crops(n, tmp_output)
        assert path.exists()
        img = Image.open(path)
        assert img.width > 0 and img.height > 0


# ---------------------------------------------------------------------------
# 6. inspect_tracks integration (real data, read-only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason="Real Phase 3 data not available locally",
)
class TestInspectTracksReal:
    def test_track_29_crops_found(self):
        with open(REAL_META) as fh:
            meta = json.load(fh)
        records = find_crops_for_track(meta, 29, _REPO_ROOT)
        assert len(records) == 5

    def test_track_66_crops_found(self):
        with open(REAL_META) as fh:
            meta = json.load(fh)
        records = find_crops_for_track(meta, 66, _REPO_ROOT)
        assert len(records) == 5

    def test_track_29_all_files_exist(self):
        with open(REAL_META) as fh:
            meta = json.load(fh)
        records = find_crops_for_track(meta, 29, _REPO_ROOT)
        for rec in records:
            assert rec.full_path.exists(), f"Missing: {rec.full_path}"

    def test_track_66_all_files_exist(self):
        with open(REAL_META) as fh:
            meta = json.load(fh)
        records = find_crops_for_track(meta, 66, _REPO_ROOT)
        for rec in records:
            assert rec.full_path.exists(), f"Missing: {rec.full_path}"

    def test_all_track_29_crops_readable(self):
        with open(REAL_META) as fh:
            meta = json.load(fh)
        records = find_crops_for_track(meta, 29, _REPO_ROOT)
        for rec in records:
            result = load_crop(rec)
            assert result.image is not None, (
                f"Could not load {rec.crop_path}: {result.error}"
            )

    def test_all_track_66_crops_readable(self):
        with open(REAL_META) as fh:
            meta = json.load(fh)
        records = find_crops_for_track(meta, 66, _REPO_ROOT)
        for rec in records:
            result = load_crop(rec)
            assert result.image is not None, (
                f"Could not load {rec.crop_path}: {result.error}"
            )

    def test_contact_sheets_generated(self, tmp_output):
        saved = inspect_tracks(
            track_ids=[29, 66],
            output_dir=tmp_output,
            source_track=29,
        )
        assert 29 in saved
        assert 66 in saved
        assert saved[29].exists()
        assert saved[66].exists()

    def test_contact_sheets_are_readable_jpegs(self, tmp_output):
        saved = inspect_tracks(
            track_ids=[29, 66],
            output_dir=tmp_output,
            source_track=29,
        )
        for tid, path in saved.items():
            img = Image.open(path)
            assert img.format == "JPEG", f"Track {tid}: expected JPEG"
            assert img.mode == "RGB",    f"Track {tid}: expected RGB"

    def test_contact_sheets_not_empty(self, tmp_output):
        saved = inspect_tracks(
            track_ids=[29, 66],
            output_dir=tmp_output,
            source_track=29,
        )
        for tid, path in saved.items():
            size = path.stat().st_size
            assert size > 5000, f"Track {tid}: contact sheet suspiciously small ({size} bytes)"

    def test_original_crops_not_modified(self):
        """Read a known crop, run the pipeline, confirm it is unchanged."""
        with open(REAL_META) as fh:
            meta = json.load(fh)
        records = find_crops_for_track(meta, 29, _REPO_ROOT)
        # Record pixel hash of first crop before
        first = records[0]
        img_before = Image.open(first.full_path).convert("RGB")
        px_before = list(img_before.getdata())
        # Run pipeline (into temp dir)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            inspect_tracks([29], output_dir=Path(td), source_track=29)
        # Verify crop unchanged
        img_after = Image.open(first.full_path).convert("RGB")
        px_after = list(img_after.getdata())
        assert px_before == px_after, "Source crop was modified — this must not happen"
