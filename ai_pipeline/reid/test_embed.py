"""
test_embed.py — Phase 4.3 + Phase 4.6
==============================================
Section A (Phase 4.3): Post-run validation tests for dataset/embeddings_C01.json.
  Validates the generated embeddings file WITHOUT requiring GPU or torchreid.
  Run AFTER embed.py has successfully processed all 353 crops.
  Auto-skips when the embeddings file is not available locally
  (e.g. the file lives on Google Colab / Google Drive).

Section B (Phase 4.6): Edge-case handling unit tests.
  Tests validate_and_preprocess() in isolation and the full pipeline via a
  lightweight mock model.  Does NOT require GPU, torchreid, or the real dataset.

Run:
    pytest ai_pipeline/reid/test_embed.py -v

    # Or with explicit paths (override defaults):
    pytest ai_pipeline/reid/test_embed.py -v \\
        --embeddings dataset/embeddings_C01.json \\
        --metadata   dataset/crops_metadata_phase3_final.json
"""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Repo root and defaults
# ---------------------------------------------------------------------------

_REPO_ROOT         = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS = _REPO_ROOT / "dataset" / "embeddings_C01.json"
DEFAULT_METADATA   = _REPO_ROOT / "dataset" / "crops_metadata_phase3_final.json"

# ---------------------------------------------------------------------------
# Constants (Phase 4.3)
# ---------------------------------------------------------------------------

EXPECTED_EMBEDDING_DIM = 512
EXPECTED_RECORD_COUNT  = 353

REQUIRED_FIELDS = (
    "crop_path",
    "camera_id",
    "track_id",
    "frame",
    "timestamp",
    "bbox",
    "detection_confidence",
    "embedding",
)

# ---------------------------------------------------------------------------
# Session-scoped fixtures — load files once, share across all Section A tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embeddings_path(request) -> Path:
    return Path(request.config.getoption("--embeddings"))


@pytest.fixture(scope="session")
def metadata_path(request) -> Path:
    return Path(request.config.getoption("--metadata"))


@pytest.fixture(scope="session")
def embeddings(embeddings_path: Path) -> list[dict]:
    """
    Load and return the full embeddings JSON list.
    Auto-skips the entire Section A suite when the file doesn't exist locally.
    """
    if not embeddings_path.exists():
        pytest.skip(
            f"embeddings_C01.json not found locally ({embeddings_path}). "
            "Run embed.py first, or run this test suite in Google Colab "
            "against the real dataset."
        )
    with open(embeddings_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def metadata(metadata_path: Path) -> list[dict]:
    """
    Load and return the Phase 3 metadata JSON list.
    Auto-skips when the metadata file doesn't exist locally.
    """
    if not metadata_path.exists():
        pytest.skip(
            f"Phase 3 metadata not found locally ({metadata_path}). "
            "Ensure the dataset is present before running Section A tests."
        )
    with open(metadata_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def embeddings_by_path(embeddings: list[dict]) -> dict[str, dict]:
    """Index embeddings by crop_path for O(1) lookup."""
    return {rec["crop_path"]: rec for rec in embeddings}


@pytest.fixture(scope="session")
def metadata_paths(metadata: list[dict]) -> set[str]:
    """Set of crop_path values from the Phase 3 metadata."""
    return {rec["crop_path"] for rec in metadata}


# ============================================================================
# SECTION A — Phase 4.3: Real embeddings_C01.json validation
# (auto-skipped when embeddings_C01.json is absent from local disk)
# ============================================================================

class TestFileExists:
    def test_embeddings_file_exists(self, embeddings_path: Path):
        if not embeddings_path.exists():
            pytest.skip("embeddings_C01.json not available locally")
        assert embeddings_path.exists()


class TestJsonLoads:
    def test_json_is_valid(self, embeddings_path: Path):
        if not embeddings_path.exists():
            pytest.skip("embeddings_C01.json not available locally")
        with open(embeddings_path) as f:
            data = json.load(f)
        assert isinstance(data, list), (
            f"Expected a JSON list at top level, got {type(data).__name__}"
        )

    def test_json_is_not_empty(self, embeddings: list[dict]):
        assert len(embeddings) > 0, "Embeddings file is empty"


class TestRecordCount:
    def test_record_count_equals_expected(self, embeddings: list[dict]):
        assert len(embeddings) == EXPECTED_RECORD_COUNT, (
            f"Expected {EXPECTED_RECORD_COUNT} records, "
            f"got {len(embeddings)}"
        )


class TestCropFilesExist:
    def test_all_crop_files_exist(self, embeddings: list[dict]):
        missing = []
        for rec in embeddings:
            p = _REPO_ROOT / rec["crop_path"]
            if not p.exists():
                missing.append(rec["crop_path"])
        assert not missing, (
            f"{len(missing)} crop file(s) referenced in embeddings do not exist:\n"
            + "\n".join(f"  {p}" for p in missing[:10])
            + ("\n  ..." if len(missing) > 10 else "")
        )


class TestNoDuplicates:
    def test_no_duplicate_crop_paths(self, embeddings: list[dict]):
        paths = [rec["crop_path"] for rec in embeddings]
        seen: set[str] = set()
        duplicates: list[str] = []
        for p in paths:
            if p in seen:
                duplicates.append(p)
            seen.add(p)
        assert not duplicates, (
            f"{len(duplicates)} duplicate crop_path value(s) found:\n"
            + "\n".join(f"  {p}" for p in duplicates[:10])
        )


class TestRequiredFields:
    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_field_present_in_all_records(self, embeddings: list[dict], field: str):
        missing_in = [
            rec.get("crop_path", f"<record #{i}>")
            for i, rec in enumerate(embeddings)
            if field not in rec
        ]
        assert not missing_in, (
            f"Field '{field}' missing in {len(missing_in)} record(s):\n"
            + "\n".join(f"  {p}" for p in missing_in[:5])
        )


class TestEmbeddingType:
    def test_embedding_is_list_in_all_records(self, embeddings: list[dict]):
        bad = [
            rec["crop_path"]
            for rec in embeddings
            if not isinstance(rec.get("embedding"), list)
        ]
        assert not bad, (
            f"'embedding' is not a list in {len(bad)} record(s):\n"
            + "\n".join(f"  {p}" for p in bad[:5])
        )


class TestEmbeddingLength:
    def test_embedding_length_is_512_in_all_records(self, embeddings: list[dict]):
        wrong = [
            (rec["crop_path"], len(rec["embedding"]))
            for rec in embeddings
            if isinstance(rec.get("embedding"), list)
            and len(rec["embedding"]) != EXPECTED_EMBEDDING_DIM
        ]
        assert not wrong, (
            f"Wrong embedding length in {len(wrong)} record(s) "
            f"(expected {EXPECTED_EMBEDDING_DIM}):\n"
            + "\n".join(f"  {p}: got {n}" for p, n in wrong[:5])
        )


class TestEmbeddingNumeric:
    def test_embedding_values_are_numeric(self, embeddings: list[dict]):
        bad = []
        for rec in embeddings:
            emb = rec.get("embedding")
            if not isinstance(emb, list):
                continue
            non_numeric = [
                (i, v) for i, v in enumerate(emb)
                if not isinstance(v, (int, float))
            ]
            if non_numeric:
                bad.append((rec["crop_path"], non_numeric[:3]))
        assert not bad, (
            f"Non-numeric embedding values in {len(bad)} record(s):\n"
            + "\n".join(f"  {p}: indices {idxs}" for p, idxs in bad[:5])
        )


class TestNoNaN:
    def test_no_nan_in_any_embedding(self, embeddings: list[dict]):
        bad = []
        for rec in embeddings:
            emb = rec.get("embedding")
            if not isinstance(emb, list):
                continue
            nan_indices = [
                i for i, v in enumerate(emb)
                if isinstance(v, float) and math.isnan(v)
            ]
            if nan_indices:
                bad.append((rec["crop_path"], nan_indices[:3]))
        assert not bad, (
            f"NaN values found in {len(bad)} embedding(s):\n"
            + "\n".join(f"  {p}: NaN at indices {idxs}" for p, idxs in bad[:5])
        )


class TestNoInf:
    def test_no_inf_in_any_embedding(self, embeddings: list[dict]):
        bad = []
        for rec in embeddings:
            emb = rec.get("embedding")
            if not isinstance(emb, list):
                continue
            inf_indices = [
                i for i, v in enumerate(emb)
                if isinstance(v, float) and math.isinf(v)
            ]
            if inf_indices:
                bad.append((rec["crop_path"], inf_indices[:3]))
        assert not bad, (
            f"Infinite values found in {len(bad)} embedding(s):\n"
            + "\n".join(f"  {p}: Inf at indices {idxs}" for p, idxs in bad[:5])
        )


class TestNoZeroEmbeddings:
    def test_no_all_zero_embeddings(self, embeddings: list[dict]):
        zero_records = [
            rec["crop_path"]
            for rec in embeddings
            if isinstance(rec.get("embedding"), list)
            and all(v == 0.0 for v in rec["embedding"])
        ]
        assert not zero_records, (
            f"All-zero embeddings found in {len(zero_records)} record(s):\n"
            + "\n".join(f"  {p}" for p in zero_records[:5])
        )


class TestMetadataAlignment:
    def test_every_input_crop_has_an_embedding(
        self, metadata_paths: set[str], embeddings_by_path: dict[str, dict]
    ):
        missing = [p for p in metadata_paths if p not in embeddings_by_path]
        assert not missing, (
            f"{len(missing)} input crop(s) have no embedding:\n"
            + "\n".join(f"  {p}" for p in sorted(missing)[:10])
            + ("\n  ..." if len(missing) > 10 else "")
        )

    def test_no_extra_embeddings_beyond_input_set(
        self, metadata_paths: set[str], embeddings_by_path: dict[str, dict]
    ):
        extra = [p for p in embeddings_by_path if p not in metadata_paths]
        assert not extra, (
            f"{len(extra)} extra embedding(s) for crops not in the input metadata:\n"
            + "\n".join(f"  {p}" for p in sorted(extra)[:10])
        )

    def test_embedding_count_equals_metadata_count(
        self, embeddings: list[dict], metadata: list[dict]
    ):
        assert len(embeddings) == len(metadata), (
            f"Embedding count ({len(embeddings)}) != metadata count ({len(metadata)})"
        )

    def test_camera_ids_preserved(
        self, metadata: list[dict], embeddings_by_path: dict[str, dict]
    ):
        mismatches = []
        for rec in metadata:
            emb_rec = embeddings_by_path.get(rec["crop_path"])
            if emb_rec and emb_rec.get("camera_id") != rec["camera_id"]:
                mismatches.append(
                    (rec["crop_path"], rec["camera_id"], emb_rec["camera_id"])
                )
        assert not mismatches, (
            f"camera_id mismatch in {len(mismatches)} record(s):\n"
            + "\n".join(
                f"  {p}: metadata={orig} embedding={got}"
                for p, orig, got in mismatches[:5]
            )
        )

    def test_track_ids_preserved(
        self, metadata: list[dict], embeddings_by_path: dict[str, dict]
    ):
        mismatches = []
        for rec in metadata:
            emb_rec = embeddings_by_path.get(rec["crop_path"])
            if emb_rec and emb_rec.get("track_id") != rec["track_id"]:
                mismatches.append(
                    (rec["crop_path"], rec["track_id"], emb_rec["track_id"])
                )
        assert not mismatches, (
            f"track_id mismatch in {len(mismatches)} record(s):\n"
            + "\n".join(
                f"  {p}: metadata={orig} embedding={got}"
                for p, orig, got in mismatches[:5]
            )
        )


# ============================================================================
# SECTION B — Phase 4.6: Edge-case handling unit tests
# ============================================================================
#
# No GPU, no torchreid, no real dataset files touched.
#
# Path resolution note
# --------------------
# run_embedding_pipeline() resolves crop paths as:
#
#     img_path = _REPO_ROOT / rec["crop_path"]
#
# In the pipeline integration tests we patch _REPO_ROOT to Path("") so that
# `Path("") / absolute_path` == `absolute_path`.  This lets us use absolute
# tmp_path values as crop_path strings without needing the files to live
# anywhere near the real repo.

MIN_W   = 40    # matches config.yaml reid.min_crop_width
MIN_H   = 100   # matches config.yaml reid.min_crop_height
EMB_DIM = 8     # small fake dimension — avoids 512-element mock tensors


# ---------------------------------------------------------------------------
# Helpers — create test images
# ---------------------------------------------------------------------------

def _make_valid_jpg(path: Path, width: int = 80, height: int = 200) -> None:
    """Write a real JPEG large enough to pass size checks."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(path, format="JPEG")


def _make_broken_jpg(path: Path) -> None:
    """Write a file with a JPEG magic number but an invalid body."""
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 8 + b"NOT A REAL JPEG BODY")


def _make_tiny_jpg(path: Path) -> None:
    """Write a valid JPEG that is too small (10×20 — below both thresholds)."""
    img = Image.new("RGB", (10, 20), color=(200, 100, 50))
    img.save(path, format="JPEG")


def _make_metadata_record(crop_path: str, **overrides) -> dict:
    """Return a minimal metadata record."""
    base = {
        "crop_path":            crop_path,
        "camera_id":            "C01",
        "track_id":             1,
        "frame":                100,
        "timestamp":            "10:02:00.00",
        "bbox":                 [10.0, 20.0, 90.0, 220.0],
        "detection_confidence": 0.95,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Mock model factory
# ---------------------------------------------------------------------------

def _make_mock_model(emb_dim: int = EMB_DIM) -> MagicMock:
    """
    Mock nn.Module whose __call__ returns a [B, emb_dim] float32 tensor.
    eval() and to() return self so the pipeline's chained calls work.
    """
    model = MagicMock()
    model.eval.return_value = model
    model.to.return_value   = model

    def _forward(batch_tensor: torch.Tensor) -> torch.Tensor:
        b = batch_tensor.shape[0]
        return torch.arange(b * emb_dim, dtype=torch.float32).reshape(b, emb_dim) + 1.0

    model.side_effect = _forward
    return model


# ---------------------------------------------------------------------------
# Import symbols under test
# ---------------------------------------------------------------------------

from ai_pipeline.reid.embed import (
    CropFailure,
    REASON_CROP_TOO_SMALL,
    REASON_INVALID_IMAGE,
    REASON_MISSING_FILE,
    REASON_PREPROCESSING_ERROR,
    REASON_UNREADABLE_IMAGE,
    run_embedding_pipeline,
    validate_and_preprocess,
)


# ============================================================================
# B.1 — validate_and_preprocess unit tests (no pipeline, no model)
# ============================================================================

class TestValidateAndPreprocess:
    """Unit tests for validate_and_preprocess(). No GPU, no model."""

    # Test 1 — corrupted JPEG
    def test_corrupted_jpeg_returns_failure(self, tmp_path: Path):
        """Edge case 1: PIL-undecodable file → unreadable_image failure."""
        p = tmp_path / "broken.jpg"
        _make_broken_jpg(p)

        tensor, failure = validate_and_preprocess(
            image_path=p,
            crop_path_str=str(p),
            min_crop_width=MIN_W,
            min_crop_height=MIN_H,
        )

        assert tensor is None, "Corrupted image must return None tensor"
        assert isinstance(failure, CropFailure)
        assert failure.reason == REASON_UNREADABLE_IMAGE
        assert failure.crop_path == str(p)

    # Test 2 — missing file
    def test_missing_file_returns_failure(self, tmp_path: Path):
        """Edge case 2: non-existent path → missing_file failure."""
        p = tmp_path / "does_not_exist.jpg"
        assert not p.exists()

        tensor, failure = validate_and_preprocess(
            image_path=p,
            crop_path_str=str(p),
            min_crop_width=MIN_W,
            min_crop_height=MIN_H,
        )

        assert tensor is None
        assert isinstance(failure, CropFailure)
        assert failure.reason == REASON_MISSING_FILE

    # Test 3 — crop below minimum width
    def test_crop_below_min_width_returns_failure(self, tmp_path: Path):
        """Edge case 3a: width < min_crop_width → crop_too_small failure."""
        p = tmp_path / "narrow.jpg"
        Image.new("RGB", (10, 200)).save(p, format="JPEG")   # w=10, h=200

        tensor, failure = validate_and_preprocess(
            image_path=p,
            crop_path_str=str(p),
            min_crop_width=MIN_W,
            min_crop_height=MIN_H,
        )

        assert tensor is None
        assert isinstance(failure, CropFailure)
        assert failure.reason == REASON_CROP_TOO_SMALL
        assert "10"      in failure.details   # actual width
        assert str(MIN_W) in failure.details  # threshold

    # Test 4 — crop below minimum height
    def test_crop_below_min_height_returns_failure(self, tmp_path: Path):
        """Edge case 3b: height < min_crop_height → crop_too_small failure."""
        p = tmp_path / "short.jpg"
        Image.new("RGB", (80, 20)).save(p, format="JPEG")    # w=80, h=20

        tensor, failure = validate_and_preprocess(
            image_path=p,
            crop_path_str=str(p),
            min_crop_width=MIN_W,
            min_crop_height=MIN_H,
        )

        assert tensor is None
        assert isinstance(failure, CropFailure)
        assert failure.reason == REASON_CROP_TOO_SMALL
        assert "20"       in failure.details   # actual height
        assert str(MIN_H) in failure.details   # threshold

    # Test 5 — valid crop after corrupted crop
    def test_valid_crop_after_corrupted_crop(self, tmp_path: Path):
        """
        A valid image must succeed even after a corrupted-image call.
        Failures are fully isolated to the individual crop.
        """
        broken = tmp_path / "broken.jpg"
        valid  = tmp_path / "valid.jpg"
        _make_broken_jpg(broken)
        _make_valid_jpg(valid)

        t1, f1 = validate_and_preprocess(
            image_path=broken, crop_path_str=str(broken),
            min_crop_width=MIN_W, min_crop_height=MIN_H,
        )
        assert t1 is None
        assert f1 is not None

        t2, f2 = validate_and_preprocess(
            image_path=valid, crop_path_str=str(valid),
            min_crop_width=MIN_W, min_crop_height=MIN_H,
        )
        assert f2 is None, "Valid crop must not fail"
        assert isinstance(t2, torch.Tensor)
        assert t2.shape == (3, 256, 128)

    # Valid crop returns correct tensor
    def test_valid_crop_returns_tensor(self, tmp_path: Path):
        """A properly-sized valid JPEG returns a [3, 256, 128] float32 tensor."""
        p = tmp_path / "valid.jpg"
        _make_valid_jpg(p, width=80, height=200)

        tensor, failure = validate_and_preprocess(
            image_path=p, crop_path_str=str(p),
            min_crop_width=MIN_W, min_crop_height=MIN_H,
        )

        assert failure is None
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 256, 128)
        assert tensor.dtype == torch.float32

    # CropFailure.to_dict() contract
    def test_failure_to_dict_has_required_keys(self, tmp_path: Path):
        """Failure serialises to a dict with crop_path / reason / details."""
        p = tmp_path / "missing.jpg"
        _, failure = validate_and_preprocess(
            image_path=p, crop_path_str="dataset/crops/missing.jpg",
            min_crop_width=MIN_W, min_crop_height=MIN_H,
        )
        d = failure.to_dict()
        assert "crop_path" in d
        assert "reason"    in d
        assert "details"   in d
        assert d["crop_path"] == "dataset/crops/missing.jpg"
        assert d["reason"]    == REASON_MISSING_FILE


# ============================================================================
# B.2 — Pipeline integration tests (mock model, no GPU, no real dataset)
# ============================================================================
#
# Path resolution: run_embedding_pipeline does:
#
#     img_path = _REPO_ROOT / rec["crop_path"]
#
# We patch ai_pipeline.reid.embed._REPO_ROOT = Path("") so that:
#
#     Path("") / "/absolute/path/to/tmp/valid.jpg"
#     == Path("/absolute/path/to/tmp/valid.jpg")
#
# This lets us store absolute tmp_path-based paths as crop_path strings.

class TestPipelineEdgeCases:
    """Full pipeline tests with mock model. No GPU, no torchreid."""

    def _run_pipeline(
        self,
        tmp_path: Path,
        records: list[dict],
        *,
        batch_size: int = 4,
    ) -> tuple[list[dict], int]:
        """
        Write metadata, run the pipeline with a patched model and _REPO_ROOT,
        return (results_list, n_failed).
        """
        meta_file   = tmp_path / "metadata.json"
        output_file = tmp_path / "embeddings.json"
        with open(meta_file, "w") as f:
            json.dump(records, f)

        mock_model = _make_mock_model(EMB_DIM)

        with patch("ai_pipeline.reid.embed.load_model", return_value=mock_model), \
             patch("ai_pipeline.reid.embed._REPO_ROOT", new=Path("")):
            n_failed = run_embedding_pipeline(
                metadata_path=meta_file,
                crop_dir=tmp_path,       # crop_dir existence check only
                output_path=output_file,
                batch_size=batch_size,
                model_name="osnet_x1_0",
                expected_dim=EMB_DIM,
                min_crop_width=MIN_W,
                min_crop_height=MIN_H,
                device=torch.device("cpu"),
                overwrite=True,
            )

        with open(output_file) as f:
            results = json.load(f)
        return results, n_failed

    # Test 12 + 7 + 9 + 10 — core pipeline resilience
    def test_pipeline_continues_after_broken_crop(self, tmp_path: Path):
        """
        Critical integration test:
            valid_1.jpg  → embedded
            broken.jpg   → logged/skipped
            valid_2.jpg  → embedded
        The pipeline MUST NOT crash.
        """
        valid_1 = tmp_path / "valid_1.jpg"
        broken  = tmp_path / "broken.jpg"
        valid_2 = tmp_path / "valid_2.jpg"

        _make_valid_jpg(valid_1)
        _make_broken_jpg(broken)
        _make_valid_jpg(valid_2)

        records = [
            _make_metadata_record(str(valid_1), track_id=1),
            _make_metadata_record(str(broken),  track_id=2),
            _make_metadata_record(str(valid_2), track_id=3),
        ]

        results, n_failed = self._run_pipeline(tmp_path, records)
        embedded_paths = {r["crop_path"] for r in results}

        # Test 7: failure was logged (n_failed > 0)
        assert n_failed == 1, f"Expected 1 failure, got {n_failed}"

        # Test 9: corrupted crop absent from successful embeddings
        assert str(broken) not in embedded_paths, (
            "Corrupted crop must not appear in successful embeddings"
        )

        # Valid crops received embeddings
        assert str(valid_1) in embedded_paths
        assert str(valid_2) in embedded_paths

        # Test 10: total = successes + failures
        assert len(results) + n_failed == 3

    # Test 5 (pipeline level) — isolation of failure
    def test_valid_crop_gets_embedding_after_corrupted_crop(self, tmp_path: Path):
        """Crop immediately after a corrupted one still gets embedded."""
        broken = tmp_path / "broken.jpg"
        valid  = tmp_path / "valid.jpg"
        _make_broken_jpg(broken)
        _make_valid_jpg(valid)

        records = [
            _make_metadata_record(str(broken), track_id=1),
            _make_metadata_record(str(valid),  track_id=2),
        ]

        results, n_failed = self._run_pipeline(tmp_path, records)

        assert len(results) == 1, "Exactly one valid crop should be embedded"
        assert results[0]["crop_path"] == str(valid)
        assert n_failed == 1

    # Test 6 — multiple valid + one corrupted
    def test_multiple_valid_crops_with_one_corrupted(self, tmp_path: Path):
        """5 valid + 1 corrupted → 5 embeddings, 1 failure."""
        n_valid = 5
        valid_paths = [tmp_path / f"valid_{i}.jpg" for i in range(n_valid)]
        broken      = tmp_path / "broken.jpg"

        for p in valid_paths:
            _make_valid_jpg(p)
        _make_broken_jpg(broken)

        records = [
            _make_metadata_record(str(p), track_id=i)
            for i, p in enumerate(valid_paths)
        ]
        records.append(_make_metadata_record(str(broken), track_id=99))

        results, n_failed = self._run_pipeline(tmp_path, records)

        assert len(results) == n_valid
        assert n_failed == 1
        assert len(results) + n_failed == n_valid + 1
        assert str(broken) not in {r["crop_path"] for r in results}

    # Test 2 (pipeline level) — missing file
    def test_missing_file_handled_in_pipeline(self, tmp_path: Path):
        """Missing file is logged; other valid crops still embedded."""
        missing = tmp_path / "missing.jpg"    # intentionally NOT created
        valid   = tmp_path / "valid.jpg"
        _make_valid_jpg(valid)

        records = [
            _make_metadata_record(str(missing), track_id=1),
            _make_metadata_record(str(valid),   track_id=2),
        ]

        results, n_failed = self._run_pipeline(tmp_path, records)

        assert n_failed == 1
        assert len(results) == 1
        assert results[0]["crop_path"] == str(valid)

    # Tests 3 + 4 (pipeline level) — too-small crops
    def test_small_crop_handled_in_pipeline(self, tmp_path: Path):
        """
        valid.jpg (80×200) → embedded
        tiny.jpg  (10×20)  → skipped (crop_too_small)
        """
        valid = tmp_path / "valid.jpg"
        tiny  = tmp_path / "tiny.jpg"
        _make_valid_jpg(valid, width=80, height=200)
        _make_tiny_jpg(tiny)

        records = [
            _make_metadata_record(str(valid), track_id=1),
            _make_metadata_record(str(tiny),  track_id=2),
        ]

        results, n_failed = self._run_pipeline(tmp_path, records)

        assert n_failed == 1
        assert len(results) == 1
        assert results[0]["crop_path"] == str(valid)

    # Critical integration test — spec requirement
    def test_critical_integration_valid_broken_tiny(self, tmp_path: Path):
        """
        Critical integration test matching the spec requirement:
            valid_1.jpg  → embedded
            valid_2.jpg  → embedded
            broken.jpg   → skipped (unreadable_image)
            tiny.jpg     → skipped (crop_too_small)

        Total: 4  |  Successful: 2  |  Failed: 2
        Pipeline MUST NOT crash.
        """
        valid_1 = tmp_path / "valid_1.jpg"
        valid_2 = tmp_path / "valid_2.jpg"
        broken  = tmp_path / "broken.jpg"
        tiny    = tmp_path / "tiny.jpg"

        _make_valid_jpg(valid_1)
        _make_valid_jpg(valid_2)
        _make_broken_jpg(broken)
        _make_tiny_jpg(tiny)

        records = [
            _make_metadata_record(str(valid_1), track_id=1),
            _make_metadata_record(str(valid_2), track_id=2),
            _make_metadata_record(str(broken),  track_id=3),
            _make_metadata_record(str(tiny),    track_id=4),
        ]

        results, n_failed = self._run_pipeline(tmp_path, records)
        n_success     = len(results)
        embedded_paths = {r["crop_path"] for r in results}

        assert n_success == 2, f"Expected 2 successful embeddings, got {n_success}"
        assert n_failed  == 2, f"Expected 2 failures, got {n_failed}"
        assert n_success + n_failed == 4

        assert str(valid_1) in embedded_paths
        assert str(valid_2) in embedded_paths
        assert str(broken)  not in embedded_paths
        assert str(tiny)    not in embedded_paths

    # Tests 8 + 11 — schema correctness of successful records
    def test_valid_crops_receive_correct_schema(self, tmp_path: Path):
        """
        Test 8 + 11: each successful embedding record has all required
        fields and a list-type embedding with the correct length.
        """
        valid = tmp_path / "valid.jpg"
        _make_valid_jpg(valid)

        records = [_make_metadata_record(str(valid), track_id=7, camera_id="C99")]

        results, _ = self._run_pipeline(tmp_path, records)

        assert len(results) == 1
        rec = results[0]

        for field in REQUIRED_FIELDS:
            assert field in rec, f"Missing required field: {field}"

        assert isinstance(rec["embedding"], list)
        assert len(rec["embedding"]) == EMB_DIM
        assert rec["camera_id"] == "C99"
        assert rec["track_id"]  == 7

        for v in rec["embedding"]:
            assert isinstance(v, (int, float))
            assert not math.isnan(v)
            assert not math.isinf(v)
