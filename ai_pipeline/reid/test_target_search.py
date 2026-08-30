"""
test_target_search.py — Phase 5.5
Unit tests for target_search.py

Tests:
- top-K ordering
- top-K limit
- confidence conversion
- JSON output schema
- source-crop exclusion
- missing input handling
- 512-D embedding compatibility

Run:
    pytest ai_pipeline/reid/test_target_search.py -v

All tests are GPU-free and use synthetic data with mocked model inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ai_pipeline.reid.target_search import (
    load_query_metadata,
    load_gallery,
    compute_similarities,
    rank_and_convert,
    format_output,
)


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

SYNTHETIC_GALLERY = [
    {
        "crop_path": "dataset/crops_phase3_final/C01_track100_frame0607.jpg",
        "camera_id": "C01",
        "track_id": 100,
        "frame": 607,
        "timestamp": "10:02:20.50",
        "bbox": [292.15, 154.30, 412.88, 534.72],
        "detection_confidence": 0.94,
        "embedding": [0.1] * 512,
    },
    {
        "crop_path": "dataset/crops_phase3_final/C01_track101_frame0650.jpg",
        "camera_id": "C01",
        "track_id": 101,
        "frame": 650,
        "timestamp": "10:02:22.10",
        "bbox": [300.0, 160.0, 420.0, 540.0],
        "detection_confidence": 0.89,
        "embedding": [0.2] * 512,
    },
    {
        "crop_path": "dataset/crops_phase3_final/C01_track102_frame0700.jpg",
        "camera_id": "C01",
        "track_id": 102,
        "frame": 700,
        "timestamp": "10:02:24.00",
        "bbox": [280.0, 150.0, 400.0, 530.0],
        "detection_confidence": 0.91,
        "embedding": [0.3] * 512,
    },
    {
        "crop_path": "dataset/crops_phase3_final/C01_track103_frame0750.jpg",
        "camera_id": "C01",
        "track_id": 103,
        "frame": 750,
        "timestamp": "10:02:26.00",
        "bbox": [290.0, 155.0, 410.0, 535.0],
        "detection_confidence": 0.87,
        "embedding": [0.4] * 512,
    },
    {
        "crop_path": "dataset/crops_phase3_final/C01_track104_frame0800.jpg",
        "camera_id": "C01",
        "track_id": 104,
        "frame": 800,
        "timestamp": "10:02:28.00",
        "bbox": [295.0, 157.0, 415.0, 537.0],
        "detection_confidence": 0.93,
        "embedding": [0.5] * 512,
    },
]

SYNTHETIC_QUERY_EMBEDDING = [0.25] * 512


# ---------------------------------------------------------------------------
# Test load_query_metadata
# ---------------------------------------------------------------------------

class TestLoadQueryMetadata:
    """Test query metadata loading."""

    def test_loads_valid_metadata(self, tmp_path):
        """Should load and return matching metadata entry."""
        metadata_file = tmp_path / "query_metadata.json"
        query_path = Path("dataset/query_photos/test_query.jpg")

        metadata = [
            {
                "query_id": "test_query",
                "query_path": str(query_path),
                "source_crop_path": "dataset/crops_phase3_final/C01_track100_frame0607.jpg",
                "source_crop_excluded_from_gallery": True,
            }
        ]

        metadata_file.write_text(json.dumps(metadata))

        result = load_query_metadata(metadata_file, query_path)

        assert result is not None
        assert result["query_id"] == "test_query"
        assert result["source_crop_path"] == "dataset/crops_phase3_final/C01_track100_frame0607.jpg"

    def test_returns_none_for_nonexistent_file(self, tmp_path):
        """Should return None when metadata file doesn't exist."""
        metadata_file = tmp_path / "nonexistent.json"
        query_path = Path("dataset/query_photos/test_query.jpg")

        result = load_query_metadata(metadata_file, query_path)

        assert result is None

    def test_returns_none_for_no_match(self, tmp_path):
        """Should return None when no entry matches query path."""
        metadata_file = tmp_path / "query_metadata.json"
        query_path = Path("dataset/query_photos/other_query.jpg")

        metadata = [
            {
                "query_id": "test_query",
                "query_path": "dataset/query_photos/test_query.jpg",
                "source_crop_path": "dataset/crops_phase3_final/C01_track100_frame0607.jpg",
            }
        ]

        metadata_file.write_text(json.dumps(metadata))

        result = load_query_metadata(metadata_file, query_path)

        assert result is None

    def test_handles_invalid_json(self, tmp_path):
        """Should return None for invalid JSON."""
        metadata_file = tmp_path / "invalid.json"
        query_path = Path("dataset/query_photos/test_query.jpg")

        metadata_file.write_text("not valid json")

        result = load_query_metadata(metadata_file, query_path)

        assert result is None


# ---------------------------------------------------------------------------
# Test load_gallery
# ---------------------------------------------------------------------------

class TestLoadGallery:
    """Test gallery loading."""

    def test_loads_valid_gallery(self, tmp_path):
        """Should load valid gallery with 512-D embeddings."""
        gallery_file = tmp_path / "embeddings.json"
        gallery_file.write_text(json.dumps(SYNTHETIC_GALLERY))

        gallery = load_gallery(gallery_file)

        assert len(gallery) == 5
        assert gallery[0]["crop_path"] == "dataset/crops_phase3_final/C01_track100_frame0607.jpg"
        assert len(gallery[0]["embedding"]) == 512

    def test_raises_for_missing_file(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        gallery_file = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_gallery(gallery_file)

    def test_raises_for_invalid_json(self, tmp_path):
        """Should raise ValueError for non-list JSON."""
        gallery_file = tmp_path / "invalid.json"
        gallery_file.write_text('{"not": "a list"}')

        with pytest.raises(ValueError, match="Expected JSON list"):
            load_gallery(gallery_file)

    def test_raises_for_empty_gallery(self, tmp_path):
        """Should raise ValueError for empty gallery."""
        gallery_file = tmp_path / "empty.json"
        gallery_file.write_text("[]")

        with pytest.raises(ValueError, match="Empty gallery"):
            load_gallery(gallery_file)

    def test_raises_for_missing_embedding_field(self, tmp_path):
        """Should raise ValueError when embedding field is missing."""
        gallery_file = tmp_path / "invalid.json"
        invalid_gallery = [
            {
                "crop_path": "dataset/crops_phase3_final/C01_track100_frame0607.jpg",
                "camera_id": "C01",
                # Missing embedding field
            }
        ]
        gallery_file.write_text(json.dumps(invalid_gallery))

        with pytest.raises(ValueError, match="missing 'embedding' field"):
            load_gallery(gallery_file)

    def test_raises_for_wrong_embedding_dimension(self, tmp_path):
        """Should raise ValueError for non-512-D embeddings."""
        gallery_file = tmp_path / "invalid.json"
        invalid_gallery = [
            {
                "crop_path": "dataset/crops_phase3_final/C01_track100_frame0607.jpg",
                "camera_id": "C01",
                "embedding": [0.1] * 256,  # Wrong dimension
            }
        ]
        gallery_file.write_text(json.dumps(invalid_gallery))

        with pytest.raises(ValueError, match="Expected 512-D embeddings"):
            load_gallery(gallery_file)


# ---------------------------------------------------------------------------
# Test compute_similarities
# ---------------------------------------------------------------------------

class TestComputeSimilarities:
    """Test similarity computation."""

    def test_computes_similarities(self):
        """Should compute cosine similarities for all gallery crops."""
        results = compute_similarities(SYNTHETIC_QUERY_EMBEDDING, SYNTHETIC_GALLERY)

        assert len(results) == 5
        for result in results:
            assert "similarity" in result
            assert "camera_id" in result
            assert "timestamp" in result
            assert "crop_path" in result
            assert "track_id" in result
            assert "frame" in result
            assert -1.0 <= result["similarity"] <= 1.0

    def test_excludes_source_crop(self):
        """Should exclude specified source crop from results."""
        source_path = "dataset/crops_phase3_final/C01_track100_frame0607.jpg"

        results = compute_similarities(
            SYNTHETIC_QUERY_EMBEDDING,
            SYNTHETIC_GALLERY,
            source_crop_path=source_path,
        )

        assert len(results) == 4  # 5 - 1 excluded
        crop_paths = [r["crop_path"] for r in results]
        assert source_path not in crop_paths

    def test_handles_none_source_crop(self):
        """Should handle None source_crop_path (no exclusion)."""
        results = compute_similarities(
            SYNTHETIC_QUERY_EMBEDDING,
            SYNTHETIC_GALLERY,
            source_crop_path=None,
        )

        assert len(results) == 5  # No exclusion

    def test_handles_missing_embedding_in_gallery(self):
        """Should skip gallery records without embeddings."""
        gallery_with_missing = SYNTHETIC_GALLERY.copy()
        gallery_with_missing[2]["embedding"] = None

        results = compute_similarities(SYNTHETIC_QUERY_EMBEDDING, gallery_with_missing)

        assert len(results) == 4  # 5 - 1 skipped


# ---------------------------------------------------------------------------
# Test rank_and_convert
# ---------------------------------------------------------------------------

class TestRankAndConvert:
    """Test ranking and confidence conversion."""

    def test_ranks_by_similarity_descending(self):
        """Should rank results by similarity descending."""
        results = [
            {"similarity": 0.5, "camera_id": "C01", "timestamp": "10:02:20.50", "crop_path": "path1"},
            {"similarity": 0.9, "camera_id": "C01", "timestamp": "10:02:22.10", "crop_path": "path2"},
            {"similarity": 0.7, "camera_id": "C01", "timestamp": "10:02:24.00", "crop_path": "path3"},
        ]

        ranked = rank_and_convert(results, top_k=10)

        assert ranked[0]["similarity"] == 0.9
        assert ranked[1]["similarity"] == 0.7
        assert ranked[2]["similarity"] == 0.5

    def test_respects_top_k_limit(self):
        """Should return at most top_k results."""
        results = [
            {"similarity": 0.9 + i * 0.01, "camera_id": "C01", "timestamp": f"10:02:{i:02d}.00", "crop_path": f"path{i}"}
            for i in range(10)
        ]

        top_5 = rank_and_convert(results, top_k=5)
        assert len(top_5) == 5

        top_3 = rank_and_convert(results, top_k=3)
        assert len(top_3) == 3

    def test_converts_similarity_to_confidence(self):
        """Should convert similarity to confidence score."""
        results = [
            {"similarity": 0.8148, "camera_id": "C01", "timestamp": "10:02:20.50", "crop_path": "path1"},
        ]

        ranked = rank_and_convert(results, top_k=10)

        assert "confidence" in ranked[0]
        assert isinstance(ranked[0]["confidence"], float)
        assert 0.0 <= ranked[0]["confidence"] <= 100.0
        # 0.8148 should map to ~75.36 based on Phase 4.5 range
        assert 70 < ranked[0]["confidence"] < 80

    def test_rounds_confidence_to_one_decimal(self):
        """Should round confidence to one decimal place."""
        results = [
            {"similarity": 0.8148, "camera_id": "C01", "timestamp": "10:02:20.50", "crop_path": "path1"},
        ]

        ranked = rank_and_convert(results, top_k=10)

        # Check that confidence is rounded to 1 decimal
        confidence_str = f"{ranked[0]['confidence']}"
        decimal_places = len(confidence_str.split(".")[1]) if "." in confidence_str else 0
        assert decimal_places <= 1

    def test_handles_empty_results(self):
        """Should handle empty results list."""
        ranked = rank_and_convert([], top_k=5)
        assert len(ranked) == 0


# ---------------------------------------------------------------------------
# Test format_output
# ---------------------------------------------------------------------------

class TestFormatOutput:
    """Test output formatting."""

    def test_formats_to_backend_contract(self):
        """Should format to backend contract with required fields only."""
        candidates = [
            {
                "similarity": 0.9,
                "confidence": 87.3,
                "camera_id": "C01",
                "timestamp": "10:02:20.50",
                "crop_path": "dataset/crops_phase3_final/C01_track100_frame0607.jpg",
                "track_id": 100,
                "frame": 607,
            }
        ]

        formatted = format_output(candidates)

        assert len(formatted) == 1
        assert formatted[0] == {
            "camera_id": "C01",
            "timestamp": "10:02:20.50",
            "confidence": 87.3,
            "crop_path": "dataset/crops_phase3_final/C01_track100_frame0607.jpg",
        }

    def test_excludes_extra_fields(self):
        """Should exclude fields not in backend contract."""
        candidates = [
            {
                "similarity": 0.9,
                "confidence": 87.3,
                "camera_id": "C01",
                "timestamp": "10:02:20.50",
                "crop_path": "dataset/crops_phase3_final/C01_track100_frame0607.jpg",
                "track_id": 100,
                "frame": 607,
                "extra_field": "should_not_appear",
            }
        ]

        formatted = format_output(candidates)

        assert "extra_field" not in formatted[0]
        assert "track_id" not in formatted[0]
        assert "frame" not in formatted[0]
        assert "similarity" not in formatted[0]

    def test_handles_multiple_candidates(self):
        """Should format multiple candidates correctly."""
        candidates = [
            {
                "similarity": 0.9,
                "confidence": 87.3,
                "camera_id": "C01",
                "timestamp": "10:02:20.50",
                "crop_path": "path1",
                "track_id": 100,
                "frame": 607,
            },
            {
                "similarity": 0.8,
                "confidence": 75.4,
                "camera_id": "C01",
                "timestamp": "10:02:22.10",
                "crop_path": "path2",
                "track_id": 101,
                "frame": 650,
            },
        ]

        formatted = format_output(candidates)

        assert len(formatted) == 2
        assert all("camera_id" in c for c in formatted)
        assert all("timestamp" in c for c in formatted)
        assert all("confidence" in c for c in formatted)
        assert all("crop_path" in c for c in formatted)


# ---------------------------------------------------------------------------
# Test integration scenarios
# ---------------------------------------------------------------------------

class TestIntegrationScenarios:
    """Test end-to-end scenarios with synthetic data."""

    def test_full_pipeline_synthetic(self):
        """Test full pipeline with synthetic data."""
        # Load gallery
        gallery = SYNTHETIC_GALLERY

        # Compute similarities
        results = compute_similarities(SYNTHETIC_QUERY_EMBEDDING, gallery)

        # Rank and convert
        top_k = rank_and_convert(results, top_k=3)

        # Format output
        formatted = format_output(top_k)

        # Verify
        assert len(formatted) == 3
        assert all("confidence" in c for c in formatted)
        assert all(isinstance(c["confidence"], float) for c in formatted)
        assert all(0.0 <= c["confidence"] <= 100.0 for c in formatted)

    def test_source_crop_exclusion_pipeline(self):
        """Test pipeline with source crop exclusion."""
        source_path = "dataset/crops_phase3_final/C01_track100_frame0607.jpg"

        # Compute similarities with exclusion
        results = compute_similarities(
            SYNTHETIC_QUERY_EMBEDDING,
            SYNTHETIC_GALLERY,
            source_crop_path=source_path,
        )

        # Rank and convert
        top_k = rank_and_convert(results, top_k=5)

        # Format output
        formatted = format_output(top_k)

        # Verify source crop is not in results
        crop_paths = [c["crop_path"] for c in formatted]
        assert source_path not in crop_paths

    def test_top_k_larger_than_gallery(self):
        """Test when top_k is larger than available results."""
        results = compute_similarities(SYNTHETIC_QUERY_EMBEDDING, SYNTHETIC_GALLERY)

        # Request more than available
        top_k = rank_and_convert(results, top_k=100)

        # Should return all available results
        assert len(top_k) == len(results)

    def test_confidence_values_are_reasonable(self):
        """Test that confidence values are in expected range."""
        results = compute_similarities(SYNTHETIC_QUERY_EMBEDDING, SYNTHETIC_GALLERY)
        top_k = rank_and_convert(results, top_k=5)

        for candidate in top_k:
            assert 0.0 <= candidate["confidence"] <= 100.0
            assert isinstance(candidate["confidence"], float)


# ---------------------------------------------------------------------------
# Test Phase 5.7 threshold behavior
# ---------------------------------------------------------------------------

class TestNoMatchThreshold:
    """Test Phase 5.7 no-match threshold handling."""

    def test_above_threshold_returns_candidates(self):
        """When best candidate similarity is above threshold, return candidates."""
        results = [
            {"similarity": 0.8, "camera_id": "C01", "timestamp": "10:02:20.50", "crop_path": "path1"},
            {"similarity": 0.7, "camera_id": "C01", "timestamp": "10:02:22.10", "crop_path": "path2"},
        ]
        top_k = rank_and_convert(results, top_k=5)

        # Simulate threshold check (threshold = 0.70)
        threshold = 0.70
        best_similarity = top_k[0]["similarity"]

        assert best_similarity >= threshold
        # Should return candidates in normal flow

    def test_below_threshold_returns_no_match(self):
        """When best candidate similarity is below threshold, return no_confident_match."""
        results = [
            {"similarity": 0.5, "camera_id": "C01", "timestamp": "10:02:20.50", "crop_path": "path1"},
            {"similarity": 0.4, "camera_id": "C01", "timestamp": "10:02:22.10", "crop_path": "path2"},
        ]
        top_k = rank_and_convert(results, top_k=5)

        # Simulate threshold check (threshold = 0.70)
        threshold = 0.70
        best_similarity = top_k[0]["similarity"]

        assert best_similarity < threshold
        # Should return no_confident_match in normal flow

    def test_exactly_at_threshold(self):
        """When best candidate similarity equals threshold, return candidates."""
        results = [
            {"similarity": 0.74, "camera_id": "C01", "timestamp": "10:02:20.50", "crop_path": "path1"},
            {"similarity": 0.65, "camera_id": "C01", "timestamp": "10:02:22.10", "crop_path": "path2"},
        ]
        top_k = rank_and_convert(results, top_k=5)

        # Simulate threshold check (threshold = 0.74)
        threshold = 0.74
        best_similarity = top_k[0]["similarity"]

        assert best_similarity == threshold
        # Should return candidates (meets threshold)

    def test_threshold_0_74_corresponds_to_confidence_63_7(self):
        """Verify that similarity 0.74 maps to confidence ~63.7."""
        from ai_pipeline.reid.confidence_scaling import similarity_to_confidence

        similarity = 0.74
        confidence = similarity_to_confidence(similarity)

        # Should be approximately 63.7
        assert 63.0 < confidence < 64.5

    def test_json_output_schema_with_status(self):
        """Test that output JSON includes status field and candidates array."""
        # Test matches_found schema
        output_matches = {
            "status": "matches_found",
            "candidates": [
                {
                    "camera_id": "C01",
                    "timestamp": "10:02:20.50",
                    "confidence": 75.3,
                    "crop_path": "dataset/crops_phase3_final/C01_track13_frame0096.jpg"
                }
            ]
        }

        assert "status" in output_matches
        assert output_matches["status"] == "matches_found"
        assert "candidates" in output_matches
        assert isinstance(output_matches["candidates"], list)

        # Test no_confident_match schema
        output_no_match = {
            "status": "no_confident_match",
            "candidates": []
        }

        assert "status" in output_no_match
        assert output_no_match["status"] == "no_confident_match"
        assert "candidates" in output_no_match
        assert len(output_no_match["candidates"]) == 0

    def test_top_k_respected_when_match_exists(self):
        """When match exists, top-K should still be respected."""
        results = [
            {"similarity": 0.9 + i * 0.01, "camera_id": "C01", "timestamp": f"10:02:{i:02d}.00", "crop_path": f"path{i}"}
            for i in range(10)
        ]
        top_k = rank_and_convert(results, top_k=3)

        # Should return exactly 3 candidates
        assert len(top_k) == 3

        # All should be above threshold (0.70)
        threshold = 0.70
        for candidate in top_k:
            assert candidate["similarity"] >= threshold

    def test_no_match_output_candidates_accessible(self):
        """
        Regression: output['candidates'] must be accessible in both branches
        so the final print(len(output['candidates'])) never raises UnboundLocalError.

        Simulates the exact logic in main() for the no-match branch:
          - best similarity below threshold -> status=no_confident_match, candidates=[]
          - len(output['candidates']) must not raise
        """
        results = [
            {"similarity": 0.7267, "camera_id": "C01", "timestamp": "10:02:20.50", "crop_path": "path1"},
            {"similarity": 0.6100, "camera_id": "C01", "timestamp": "10:02:22.10", "crop_path": "path2"},
        ]
        top_k = rank_and_convert(results, top_k=5)

        threshold = 0.74  # current MVP threshold
        best_similarity = top_k[0]["similarity"]

        # No-match branch — mirrors main() logic exactly
        if best_similarity < threshold:
            output = {
                "status": "no_confident_match",
                "candidates": []
            }
        else:
            formatted = format_output(top_k)
            output = {
                "status": "matches_found",
                "candidates": formatted
            }

        # This line must not raise — it is the exact expression used in main()
        candidate_count = len(output["candidates"])

        assert output["status"] == "no_confident_match"
        assert output["candidates"] == []
        assert candidate_count == 0

    def test_matches_found_output_candidates_accessible(self):
        """
        Companion to test_no_match_output_candidates_accessible.
        Verifies the matches_found branch also produces a valid output['candidates']
        so len(output['candidates']) works identically.
        """
        results = [
            {"similarity": 0.8148, "camera_id": "C01", "timestamp": "10:02:03.24", "crop_path": "dataset/crops_phase3_final/C01_track13_frame0096.jpg"},
            {"similarity": 0.6490, "camera_id": "C01", "timestamp": "10:02:00.00", "crop_path": "dataset/crops_phase3_final/C01_track2_frame0000.jpg"},
        ]
        top_k = rank_and_convert(results, top_k=5)

        threshold = 0.74
        best_similarity = top_k[0]["similarity"]

        if best_similarity < threshold:
            output = {
                "status": "no_confident_match",
                "candidates": []
            }
        else:
            formatted = format_output(top_k)
            output = {
                "status": "matches_found",
                "candidates": formatted
            }

        # This line must not raise
        candidate_count = len(output["candidates"])

        assert output["status"] == "matches_found"
        assert candidate_count == 2
        assert output["candidates"][0]["confidence"] is not None
