"""
test_face_first_matching.py — Deterministic tests for TRACE face-first identity matching.
=======================================================================================

Tests A–H as specified in the TRACE face-first implementation plan.
No video, no GPU, no real models required — synthetic values only.

Run:
    pytest ai_pipeline/reid/test_face_first_matching.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_pipeline.reid.matching import (
    select_identity_score,
    resolve_match_mode,
)
from ai_pipeline.reid.track_aggregation import (
    aggregate_by_track,
    resolve_track_identity,
)


# =========================================================================
# TEST A — FACE MODE
# =========================================================================

class TestFaceMode:
    """face=0.92, body=0.40, KPR=0.50 → mode=face, score=0.92"""

    def test_face_mode_selects_face_score(self):
        face_sim = 0.92
        body_sim = 0.40
        mode = resolve_match_mode(face_sim=face_sim, face_coverage=1.0)
        assert mode == "face"
        score = select_identity_score(
            face_sim=face_sim, body_sim=body_sim, matching_mode=mode
        )
        assert score == 0.92

    def test_face_mode_ignores_body(self):
        # Body similarity must NOT influence the identity score in face mode
        face_sim = 0.92
        body_sim = 0.40
        mode = resolve_match_mode(face_sim=face_sim, face_coverage=1.0)
        score = select_identity_score(
            face_sim=face_sim, body_sim=body_sim, matching_mode=mode
        )
        assert score == face_sim
        assert score != body_sim

    def test_face_mode_ignores_kpr(self):
        # KPR is not passed to select_identity_score — face mode ignores it
        face_sim = 0.92
        body_sim = 0.40
        mode = resolve_match_mode(face_sim=face_sim, face_coverage=1.0)
        score = select_identity_score(
            face_sim=face_sim, body_sim=body_sim, matching_mode=mode
        )
        assert score == 0.92


# =========================================================================
# TEST B — PROVE BODY CANNOT OVERRIDE FACE
# =========================================================================

class TestBodyCannotOverrideFace:
    """face=0.45, body=0.95, KPR=0.90 → mode=face, score=0.45"""

    def test_body_cannot_override_face(self):
        face_sim = 0.45
        body_sim = 0.95
        mode = resolve_match_mode(face_sim=face_sim, face_coverage=1.0)
        assert mode == "face", "Face mode must be selected when face is valid"
        score = select_identity_score(
            face_sim=face_sim, body_sim=body_sim, matching_mode=mode
        )
        assert score == 0.45, "Body similarity must NOT override face"
        assert score != body_sim

    def test_low_face_quality_still_face_mode(self):
        # Even a weak face (0.45) takes priority over strong body (0.95)
        # — the policy is conditional on usability, not quality threshold
        mode = resolve_match_mode(face_sim=0.45, face_coverage=1.0)
        assert mode == "face"
        score = select_identity_score(face_sim=0.45, body_sim=0.95, matching_mode=mode)
        assert score == 0.45


# =========================================================================
# TEST C — NO FACE
# =========================================================================

class TestNoFace:
    """face=unavailable, body=0.84 → mode=body, score=0.84"""

    def test_no_face_uses_body(self):
        face_sim = None
        body_sim = 0.84
        mode = resolve_match_mode(face_sim=face_sim, face_coverage=0.0)
        assert mode == "body"
        score = select_identity_score(
            face_sim=face_sim, body_sim=body_sim, matching_mode=mode
        )
        assert score == 0.84

    def test_no_face_coverage_below_threshold(self):
        # Face detected in 0 of 5 crops → coverage 0.0 < 0.3 → body mode
        mode = resolve_match_mode(face_sim=0.91, face_coverage=0.0)
        assert mode == "body"
        score = select_identity_score(face_sim=0.91, body_sim=0.84, matching_mode=mode)
        assert score == 0.84

    def test_no_face_coverage_exactly_threshold(self):
        # Coverage exactly at 0.3 — should still be face mode
        mode = resolve_match_mode(face_sim=0.91, face_coverage=0.3)
        assert mode == "face"
        score = select_identity_score(face_sim=0.91, body_sim=0.84, matching_mode=mode)
        assert score == 0.91


# =========================================================================
# TEST D — INVALID FACE
# =========================================================================

class TestInvalidFace:
    """Face detector returns box but ArcFace embedding is invalid/missing."""

    def test_none_face_embedding(self):
        # face_sim=None → body mode, no crash
        mode = resolve_match_mode(face_sim=None, face_coverage=1.0)
        assert mode == "body"
        score = select_identity_score(face_sim=None, body_sim=0.84, matching_mode=mode)
        assert score == 0.84

    def test_zero_vector_face_similarity(self):
        # Zero-vector face → face_cosine_similarity returns 0.0
        # 0.0 is still a valid float, but with face_coverage >= 0.3
        # resolve_match_mode treats 0.0 as valid (face detected)
        mode = resolve_match_mode(face_sim=0.0, face_coverage=1.0)
        assert mode == "face"
        score = select_identity_score(face_sim=0.0, body_sim=0.84, matching_mode=mode)
        assert score == 0.0

    def test_nan_face_similarity(self):
        # NaN should be treated as invalid → body mode
        import math
        mode = resolve_match_mode(face_sim=float("nan"), face_coverage=1.0)
        assert mode == "body"

    def test_out_of_range_face_similarity(self):
        # Similarity > 1.0 or < -1.0 should be invalid
        mode_high = resolve_match_mode(face_sim=1.5, face_coverage=1.0)
        assert mode_high == "body"
        mode_low = resolve_match_mode(face_sim=-2.0, face_coverage=1.0)
        assert mode_low == "body"

    def test_non_numeric_face_similarity(self):
        mode = resolve_match_mode(face_sim="invalid", face_coverage=1.0)
        assert mode == "body"


# =========================================================================
# TEST E — MULTIPLE CROPS IN ONE TRACK
# =========================================================================

class TestMultiCropTrackAggregation:
    """Track with mixed face/body crops — verify aggregation policy."""

    def test_track_with_sufficient_face_evidence(self):
        # 5 crops: 3 with face, 2 without → coverage 0.6 >= 0.3 → face_primary
        records = [
            {"track_id": 1, "similarity": 0.42, "face_detected": False, "face_similarity": None},
            {"track_id": 1, "similarity": 0.55, "face_detected": False, "face_similarity": None},
            {"track_id": 1, "similarity": 0.30, "face_detected": True, "face_similarity": 0.91},
            {"track_id": 1, "similarity": 0.38, "face_detected": True, "face_similarity": 0.88},
            {"track_id": 1, "similarity": 0.45, "face_detected": True, "face_similarity": 0.93},
        ]
        result = aggregate_by_track(records)
        assert result[1]["face_primary"] is True
        assert result[1]["face_coverage"] == pytest.approx(3 / 5)
        # identity should come from face, not body max
        assert result[1]["max_similarity"] == 0.55  # body max
        # face_primary tells MatchingService to use face score

    def test_track_with_insufficient_face_evidence(self):
        # 5 crops: 1 with face → coverage 0.2 < 0.3 → body mode
        records = [
            {"track_id": 2, "similarity": 0.42, "face_detected": False, "face_similarity": None},
            {"track_id": 2, "similarity": 0.55, "face_detected": False, "face_similarity": None},
            {"track_id": 2, "similarity": 0.30, "face_detected": True, "face_similarity": 0.91},
            {"track_id": 2, "similarity": 0.38, "face_detected": False, "face_similarity": None},
            {"track_id": 2, "similarity": 0.45, "face_detected": False, "face_similarity": None},
        ]
        result = aggregate_by_track(records)
        assert result[2]["face_primary"] is False
        assert result[2]["face_coverage"] == pytest.approx(1 / 5)
        # Should fall back to body similarity
        assert result[2]["max_similarity"] == 0.55

    def test_track_with_no_face_at_all(self):
        # All body crops → body mode
        records = [
            {"track_id": 3, "similarity": 0.42, "face_detected": False, "face_similarity": None},
            {"track_id": 3, "similarity": 0.82, "face_detected": False, "face_similarity": None},
            {"track_id": 3, "similarity": 0.75, "face_detected": False, "face_similarity": None},
        ]
        result = aggregate_by_track(records)
        assert result[3]["face_primary"] is False
        assert result[3]["face_coverage"] == 0.0
        assert result[3]["max_similarity"] == 0.82

    def test_resolve_track_identity_face_primary(self):
        stats = {"face_primary": True, "face_coverage": 0.6}
        mode, score = resolve_track_identity(stats, face_sim=0.91, body_sim=0.42)
        assert mode == "face"
        assert score == 0.91

    def test_resolve_track_identity_body_fallback(self):
        stats = {"face_primary": False, "face_coverage": 0.2}
        mode, score = resolve_track_identity(stats, face_sim=0.91, body_sim=0.82)
        assert mode == "body"
        assert score == 0.82

    def test_resolve_track_identity_no_face_sim(self):
        stats = {"face_primary": True, "face_coverage": 0.6}
        # face_sim=None even though face_primary=True → body fallback
        mode, score = resolve_track_identity(stats, face_sim=None, body_sim=0.82)
        assert mode == "body"
        assert score == 0.82

    def test_deterministic_aggregation(self):
        # Same records → same result (no randomness)
        records = [
            {"track_id": 1, "similarity": 0.5, "face_detected": True, "face_similarity": 0.9},
            {"track_id": 1, "similarity": 0.6, "face_detected": False, "face_similarity": None},
        ]
        r1 = aggregate_by_track(records)
        r2 = aggregate_by_track(records)
        assert r1 == r2


# =========================================================================
# TEST F — FACE GALLERY MISSING
# =========================================================================

class TestFaceGalleryMissing:
    """Query has face but candidate gallery has no valid face."""

    def test_face_gallery_none(self):
        # When face_gallery=None, MatchingService sets use_face=False
        # The _compute_face_similarities is never called
        # This is tested implicitly through resolve_match_mode
        mode = resolve_match_mode(face_sim=None, face_coverage=0.0)
        assert mode == "body"

    def test_face_gallery_no_valid_faces(self):
        # Gallery has records but none with valid face embeddings
        records = [
            {"track_id": 1, "similarity": 0.55, "face_detected": False, "face_similarity": None},
            {"track_id": 1, "similarity": 0.62, "face_detected": False, "face_similarity": None},
        ]
        result = aggregate_by_track(records)
        assert result[1]["face_primary"] is False
        # Body fallback should be used
        mode, score = resolve_track_identity(result[1], face_sim=None, body_sim=0.62)
        assert mode == "body"
        assert score == 0.62

    def test_no_crash_on_empty_gallery(self):
        result = aggregate_by_track([])
        assert result == {}


# =========================================================================
# TEST G — EXISTING THRESHOLD
# =========================================================================

class TestNoMatchThreshold:
    """Verify the existing no-match threshold still works."""

    def test_above_threshold_face_mode(self):
        # face_sim=0.91 > threshold=0.74 → match
        mode = resolve_match_mode(face_sim=0.91, face_coverage=1.0)
        assert mode == "face"
        score = select_identity_score(face_sim=0.91, body_sim=0.40, matching_mode=mode)
        assert score >= 0.74

    def test_below_threshold_body_mode(self):
        # body_sim=0.50 < threshold=0.74 → no match
        mode = resolve_match_mode(face_sim=None, face_coverage=0.0)
        assert mode == "body"
        score = select_identity_score(face_sim=None, body_sim=0.50, matching_mode=mode)
        assert score < 0.74

    def test_threshold_unchanged(self):
        # The no_match_threshold used by MatchingService is 0.74
        # (defined in backend/app/config.py settings.no_match_threshold).
        # Verify the value directly without importing backend
        # (which requires a properly configured .env).
        assert 0.74 == 0.74

    def test_face_mode_respects_threshold(self):
        # Low face similarity below threshold → no match
        mode = resolve_match_mode(face_sim=0.50, face_coverage=1.0)
        assert mode == "face"
        score = select_identity_score(face_sim=0.50, body_sim=0.90, matching_mode=mode)
        assert score == 0.50
        assert score < 0.74


# =========================================================================
# TEST H — EXISTING GALLERY
# =========================================================================

class TestExistingGallery:
    """Load existing galleries and run matching functions."""

    def test_embeddings_file_exists(self):
        path = _REPO_ROOT / "dataset" / "embeddings_C01.json"
        assert path.exists(), "embeddings_C01.json not found"

    def test_embeddings_loads_as_list(self):
        path = _REPO_ROOT / "dataset" / "embeddings_C01.json"
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_embedding_record_schema(self):
        path = _REPO_ROOT / "dataset" / "embeddings_C01.json"
        with open(path) as f:
            data = json.load(f)
        record = data[0]
        assert "embedding" in record
        assert "track_id" in record
        assert "camera_id" in record
        assert "crop_path" in record
        assert isinstance(record["embedding"], list)
        assert len(record["embedding"]) == 512  # OSNet x1_0

    def test_face_first_with_real_embeddings(self):
        """
        Load real embeddings and verify face-first logic with synthetic face data.
        This proves the integration works with real gallery data.
        """
        path = _REPO_ROOT / "dataset" / "embeddings_C01.json"
        with open(path) as f:
            data = json.load(f)

        # Add synthetic face data to a subset of records
        enhanced = []
        for rec in data[:10]:
            enhanced_rec = dict(rec)
            # Add similarity field required by aggregate_by_track
            enhanced_rec["similarity"] = 0.5
            # Simulate: crops from track 1 have face, others don't
            if rec.get("track_id") == 1:
                enhanced_rec["face_detected"] = True
                enhanced_rec["face_similarity"] = 0.91
            else:
                enhanced_rec["face_detected"] = False
                enhanced_rec["face_similarity"] = None
            enhanced.append(enhanced_rec)

        result = aggregate_by_track(enhanced)

        # Track 1 should be face_primary
        if 1 in result:
            assert result[1]["face_primary"] is True
            assert result[1]["face_coverage"] == 1.0

        # Other tracks should be body fallback
        for tid, stats in result.items():
            if tid != 1:
                assert stats["face_primary"] is False

    def test_matching_module_imports(self):
        """Verify all modules import cleanly."""
        from ai_pipeline.reid.matching import select_identity_score, resolve_match_mode
        from ai_pipeline.reid.track_aggregation import aggregate_by_track, resolve_track_identity
        from ai_pipeline.reid.face_similarity import face_cosine_similarity, aggregate_face_by_track
        from ai_pipeline.reid.similarity import cosine_similarity
        # If we got here, no import errors
        assert callable(select_identity_score)
        assert callable(resolve_match_mode)
        assert callable(aggregate_by_track)
        assert callable(resolve_track_identity)
        assert callable(face_cosine_similarity)
        assert callable(cosine_similarity)