"""
Phase 4.2 — Embedding Storage Schema Tests

Validates the official JSON embedding record schema for TRACE Re-ID.

These tests do NOT run OSNet. They do NOT require GPU or torchreid.
They validate:
  - Required fields are present
  - Field types are correct
  - Embedding is a list of 512 numeric float values
  - No NaN or infinite values in the embedding
  - JSON round-trip serialization preserves the record faithfully

Run with:
    pytest ai_pipeline/reid/test_embedding_storage.py -v
"""

import json
import math
import pytest


# ---------------------------------------------------------------------------
# Shared fixture: a single valid embedding record matching the Phase 4.2
# schema. Values are synthetic — no OSNet inference is performed.
# ---------------------------------------------------------------------------

EXPECTED_EMBEDDING_DIM = 512


def make_sample_record(embedding=None):
    """Return a minimal but fully-valid Phase 4.2 embedding record."""
    if embedding is None:
        # Synthetic values: mix of positive, zero, and small negatives — realistic
        # for an OSNet output before L2 normalisation.
        embedding = [float(i % 7) * 0.1 - 0.3 for i in range(EXPECTED_EMBEDDING_DIM)]
    return {
        "crop_path": "dataset/crops_phase3_final/C01_track100_frame0607.jpg",
        "camera_id": "C01",
        "track_id": 100,
        "frame": 607,
        "timestamp": "10:02:20.50",
        "bbox": [292.15, 154.30, 412.88, 534.72],
        "detection_confidence": 0.94,
        "embedding": embedding,
    }


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

class TestRequiredFields:
    """Every embedding record must carry all Phase 3 metadata fields plus embedding."""

    def test_has_crop_path(self):
        record = make_sample_record()
        assert "crop_path" in record

    def test_has_camera_id(self):
        record = make_sample_record()
        assert "camera_id" in record

    def test_has_track_id(self):
        record = make_sample_record()
        assert "track_id" in record

    def test_has_frame(self):
        record = make_sample_record()
        assert "frame" in record

    def test_has_timestamp(self):
        record = make_sample_record()
        assert "timestamp" in record

    def test_has_bbox(self):
        record = make_sample_record()
        assert "bbox" in record

    def test_has_detection_confidence(self):
        record = make_sample_record()
        assert "detection_confidence" in record

    def test_has_embedding(self):
        record = make_sample_record()
        assert "embedding" in record


# ---------------------------------------------------------------------------
# Field types
# ---------------------------------------------------------------------------

class TestFieldTypes:
    """Field types must match the schema definition."""

    def test_crop_path_is_string(self):
        record = make_sample_record()
        assert isinstance(record["crop_path"], str)

    def test_camera_id_is_string(self):
        record = make_sample_record()
        assert isinstance(record["camera_id"], str)

    def test_track_id_is_int(self):
        record = make_sample_record()
        assert isinstance(record["track_id"], int)

    def test_frame_is_int(self):
        record = make_sample_record()
        assert isinstance(record["frame"], int)

    def test_timestamp_is_string(self):
        record = make_sample_record()
        assert isinstance(record["timestamp"], str)

    def test_bbox_is_list(self):
        record = make_sample_record()
        assert isinstance(record["bbox"], list)

    def test_bbox_has_four_elements(self):
        record = make_sample_record()
        assert len(record["bbox"]) == 4

    def test_detection_confidence_is_float(self):
        record = make_sample_record()
        assert isinstance(record["detection_confidence"], float)

    def test_embedding_is_list(self):
        record = make_sample_record()
        assert isinstance(record["embedding"], list)


# ---------------------------------------------------------------------------
# Embedding vector properties
# ---------------------------------------------------------------------------

class TestEmbeddingVector:
    """Validate the numeric properties of the embedding vector."""

    def test_embedding_length_is_512(self):
        record = make_sample_record()
        assert len(record["embedding"]) == EXPECTED_EMBEDDING_DIM, (
            f"Expected embedding length {EXPECTED_EMBEDDING_DIM}, "
            f"got {len(record['embedding'])}"
        )

    def test_embedding_values_are_numeric(self):
        record = make_sample_record()
        for i, val in enumerate(record["embedding"]):
            assert isinstance(val, (int, float)), (
                f"embedding[{i}] = {val!r} is not numeric"
            )

    def test_embedding_has_no_nan(self):
        record = make_sample_record()
        nan_indices = [i for i, v in enumerate(record["embedding"]) if math.isnan(v)]
        assert not nan_indices, (
            f"NaN values found at embedding indices: {nan_indices}"
        )

    def test_embedding_has_no_inf(self):
        record = make_sample_record()
        inf_indices = [i for i, v in enumerate(record["embedding"]) if math.isinf(v)]
        assert not inf_indices, (
            f"Infinite values found at embedding indices: {inf_indices}"
        )

    def test_embedding_contains_nonzero_values(self):
        """A valid OSNet embedding should not be an all-zero vector."""
        record = make_sample_record()
        nonzero = [v for v in record["embedding"] if v != 0.0]
        assert len(nonzero) > 0, "Embedding is all zeros — likely a degenerate output"


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    """The record must survive JSON serialise → deserialise without data loss."""

    def test_json_serialization_succeeds(self):
        record = make_sample_record()
        serialized = json.dumps(record)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_json_deserialization_succeeds(self):
        record = make_sample_record()
        serialized = json.dumps(record)
        restored = json.loads(serialized)
        assert isinstance(restored, dict)

    def test_round_trip_preserves_all_fields(self):
        record = make_sample_record()
        restored = json.loads(json.dumps(record))
        for field in (
            "crop_path", "camera_id", "track_id", "frame",
            "timestamp", "bbox", "detection_confidence", "embedding",
        ):
            assert field in restored, f"Field '{field}' missing after JSON round-trip"

    def test_round_trip_preserves_scalar_values(self):
        record = make_sample_record()
        restored = json.loads(json.dumps(record))
        assert restored["crop_path"] == record["crop_path"]
        assert restored["camera_id"] == record["camera_id"]
        assert restored["track_id"] == record["track_id"]
        assert restored["frame"] == record["frame"]
        assert restored["timestamp"] == record["timestamp"]
        assert restored["detection_confidence"] == pytest.approx(record["detection_confidence"])

    def test_round_trip_preserves_bbox(self):
        record = make_sample_record()
        restored = json.loads(json.dumps(record))
        assert len(restored["bbox"]) == 4
        for orig, rest in zip(record["bbox"], restored["bbox"]):
            assert rest == pytest.approx(orig)

    def test_round_trip_preserves_embedding_length(self):
        record = make_sample_record()
        restored = json.loads(json.dumps(record))
        assert len(restored["embedding"]) == EXPECTED_EMBEDDING_DIM

    def test_round_trip_preserves_embedding_values(self):
        record = make_sample_record()
        restored = json.loads(json.dumps(record))
        for i, (orig, rest) in enumerate(zip(record["embedding"], restored["embedding"])):
            assert rest == pytest.approx(orig), (
                f"Embedding value mismatch at index {i}: {orig} → {rest}"
            )

    def test_round_trip_embedding_still_has_no_nan(self):
        record = make_sample_record()
        restored = json.loads(json.dumps(record))
        nan_indices = [i for i, v in enumerate(restored["embedding"]) if math.isnan(v)]
        assert not nan_indices

    def test_round_trip_embedding_still_has_no_inf(self):
        record = make_sample_record()
        restored = json.loads(json.dumps(record))
        inf_indices = [i for i, v in enumerate(restored["embedding"]) if math.isinf(v)]
        assert not inf_indices


# ---------------------------------------------------------------------------
# Schema rejection: invalid records should fail validation
# ---------------------------------------------------------------------------

def validate_embedding_record(record: dict) -> list[str]:
    """
    Validate a single embedding record against the Phase 4.2 schema.
    Returns a list of error strings (empty list = valid).

    This is the reference validator that Phase 4.3 will use.
    """
    errors = []
    required_fields = {
        "crop_path": str,
        "camera_id": str,
        "track_id": int,
        "frame": int,
        "timestamp": str,
        "bbox": list,
        "detection_confidence": float,
        "embedding": list,
    }

    for field, expected_type in required_fields.items():
        if field not in record:
            errors.append(f"Missing required field: '{field}'")
        elif not isinstance(record[field], expected_type):
            errors.append(
                f"Field '{field}' has wrong type: "
                f"expected {expected_type.__name__}, got {type(record[field]).__name__}"
            )

    if "bbox" in record and isinstance(record["bbox"], list):
        if len(record["bbox"]) != 4:
            errors.append(f"'bbox' must have 4 elements, got {len(record['bbox'])}")

    if "embedding" in record and isinstance(record["embedding"], list):
        emb = record["embedding"]
        if len(emb) != EXPECTED_EMBEDDING_DIM:
            errors.append(
                f"'embedding' length must be {EXPECTED_EMBEDDING_DIM}, got {len(emb)}"
            )
        non_numeric = [i for i, v in enumerate(emb) if not isinstance(v, (int, float))]
        if non_numeric:
            errors.append(f"Non-numeric values in embedding at indices: {non_numeric[:5]}")
        nan_vals = [i for i, v in enumerate(emb) if isinstance(v, float) and math.isnan(v)]
        if nan_vals:
            errors.append(f"NaN values in embedding at indices: {nan_vals[:5]}")
        inf_vals = [i for i, v in enumerate(emb) if isinstance(v, float) and math.isinf(v)]
        if inf_vals:
            errors.append(f"Infinite values in embedding at indices: {inf_vals[:5]}")

    return errors


class TestValidator:
    """validate_embedding_record() must accept valid records and reject invalid ones."""

    def test_valid_record_passes(self):
        record = make_sample_record()
        errors = validate_embedding_record(record)
        assert errors == [], f"Valid record unexpectedly failed: {errors}"

    def test_missing_embedding_fails(self):
        record = make_sample_record()
        del record["embedding"]
        errors = validate_embedding_record(record)
        assert any("embedding" in e for e in errors)

    def test_missing_camera_id_fails(self):
        record = make_sample_record()
        del record["camera_id"]
        errors = validate_embedding_record(record)
        assert any("camera_id" in e for e in errors)

    def test_wrong_embedding_length_fails(self):
        record = make_sample_record(embedding=[0.1] * 256)  # wrong: 256 not 512
        errors = validate_embedding_record(record)
        assert any("512" in e for e in errors)

    def test_nan_in_embedding_fails(self):
        emb = [float(i % 7) * 0.1 for i in range(EXPECTED_EMBEDDING_DIM)]
        emb[42] = float("nan")
        record = make_sample_record(embedding=emb)
        errors = validate_embedding_record(record)
        assert any("NaN" in e for e in errors)

    def test_inf_in_embedding_fails(self):
        emb = [float(i % 7) * 0.1 for i in range(EXPECTED_EMBEDDING_DIM)]
        emb[100] = float("inf")
        record = make_sample_record(embedding=emb)
        errors = validate_embedding_record(record)
        assert any("Infinite" in e or "nfinite" in e for e in errors)

    def test_wrong_track_id_type_fails(self):
        record = make_sample_record()
        record["track_id"] = "100"  # string instead of int
        errors = validate_embedding_record(record)
        assert any("track_id" in e for e in errors)
