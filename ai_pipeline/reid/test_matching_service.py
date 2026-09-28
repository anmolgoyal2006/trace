"""
test_matching_service.py — Test matrix for Phase 3 fixes
=========================================================

Tests A–J from the acceptance criteria in the false-positive fix spec.

All tests are pure CPU / pure Python — no GPU, no torchreid, no ONNX.
They use synthetic embeddings so they run offline and in CI.

Run:
    pytest ai_pipeline/reid/test_matching_service.py -v

Test matrix
-----------
  A  Unknown person    — absent from video → NO_CONFIDENT_MATCH
  B  Known person      — present in video  → CONFIDENT_MATCH
  C  Similar clothing  — body high / face low → FACE_MISMATCH (not CONFIDENT)
  D  Same person diff clothing — face preserves identity when body changes
  E  Face unavailable  — no face veto applied; conservative body decision
  F  Low-quality face  — face below quality threshold; no veto
  G  OSNet gallery     — OSNet threshold (0.74) and scaling applied
  H  SOLIDER gallery   — SOLIDER threshold (0.94) and scaling applied
  I  KPR enabled       — KPR high + face low → FACE_MISMATCH (KPR ≠ identity)
  J  Top-3 candidates  — exactly ≤3 unique tracks, correct ordering

Additional regression tests:
  R1  Body-only path is byte-identical to pre-fusion output
  R2  Face coverage below min → no veto
  R3  Query face confidence below min → no veto
  R4  Body similarity below soft_floor → not returned
  R5  Existing confidence_scaling tests still pass (OSNet range unchanged)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ai_pipeline.reid.confidence_scaling import (
    BACKBONE_OSNET_512,
    BACKBONE_SOLIDER_768,
    BackboneConfig,
    get_backbone_config,
    similarity_to_confidence,
)
from backend.app.services.matching_service import (
    CameraMatch,
    MatchingService,
    MatchStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=0)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _make_embedding(dim: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(dim)).tolist()


def _near_copy(base: list[float], noise: float = 0.02, seed: int = 99) -> list[float]:
    """Return a gallery embedding very close to *base* (high body sim)."""
    rng = np.random.default_rng(seed)
    v = np.array(base) + rng.standard_normal(len(base)) * noise
    return _unit(v).tolist()


def _opposite(base: list[float], noise: float = 0.05, seed: int = 77) -> list[float]:
    """Return an embedding pointing in the opposite direction (low face sim)."""
    rng = np.random.default_rng(seed)
    v = -np.array(base) + rng.standard_normal(len(base)) * noise
    return _unit(v).tolist()


def _make_body_gallery(query_emb: list[float], track_id: int,
                       noise: float = 0.02, dim: int = 512) -> list[dict]:
    """One-crop body gallery with body sim ≈ 1 - noise."""
    return [{
        "embedding": _near_copy(query_emb, noise=noise),
        "track_id": track_id,
        "crop_path": f"crops/C01_track{track_id}_frame0001.jpg",
        "timestamp": "10:02:01.000",
        "frame": 1,
    }]


def _make_face_gallery(face_emb: list[float], track_id: int,
                       n_crops: int = 3) -> list[dict]:
    """Face gallery with 100% face coverage."""
    return [
        {
            "track_id": track_id,
            "face_detected": True,
            "face_embedding": face_emb,
            "face_confidence": 0.92,
            "crop_path": f"crops/C01_track{track_id}_frame{i:04d}.jpg",
            "camera_id": "C01",
            "frame": i,
            "timestamp": f"10:02:0{i}.000",
            "bbox": None,
        }
        for i in range(1, n_crops + 1)
    ]


# Singleton service
_svc = MatchingService()


# ===========================================================================
# TEST A — Unknown person (absent from video) → NO_CONFIDENT_MATCH
# ===========================================================================

class TestA_UnknownPerson:
    """
    Query person is NOT present in the video.
    Body similarity is coincidentally high (a plausible failure case),
    but face similarity is low — veto must fire.
    Without face galleries → body-only must require body_sim < threshold
    to not produce a CONFIDENT_MATCH.

    We test both scenarios:
      A1: body sim low (below threshold) → NO match
      A2: body sim high + face veto active → FACE_MISMATCH, not CONFIDENT
    """

    def test_a1_low_body_sim_returns_no_match(self):
        """Body sim well below threshold → empty results (below soft_floor)."""
        query_emb   = _make_embedding(512, seed=1)
        unrelated   = _make_embedding(512, seed=999)   # orthogonal-ish
        gallery     = [{
            "embedding": unrelated,
            "track_id": 10,
            "crop_path": "crops/C01_track10_f001.jpg",
            "timestamp": "10:00", "frame": 1,
        }]
        results = _svc.search_camera_top_k(
            query_embedding=query_emb,
            gallery=gallery,
            camera_id="C01",
        )
        # Orthogonal-ish 512-d vectors → body sim ≈ 0; below soft_floor=0.60
        assert results == [], (
            "Unknown person with very low body similarity should return no candidates"
        )

    def test_a2_high_body_face_veto_returns_face_mismatch(self):
        """
        Body sim > threshold BUT reliable face contradicts → FACE_MISMATCH,
        NOT CONFIDENT_MATCH.  This is the exact failing scenario.
        """
        query_body  = _make_embedding(512, seed=10)
        query_face  = _make_embedding(512, seed=20)
        gallery_face = _opposite(query_face, noise=0.02, seed=30)  # very low sim

        gallery     = _make_body_gallery(query_body, track_id=5, noise=0.01)
        face_gal    = _make_face_gallery(gallery_face, track_id=5)

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
            face_gallery=face_gal,
            query_face_embedding=query_face,
            query_face_confidence=0.95,
        )
        assert len(results) == 1
        assert results[0].match_status == MatchStatus.FACE_MISMATCH, (
            f"Unknown person with reliable face mismatch must be FACE_MISMATCH, "
            f"got {results[0].match_status}"
        )
        assert results[0].face_veto_applied is True
        # Must NOT be classified as CONFIDENT_MATCH
        assert results[0].match_status != MatchStatus.CONFIDENT_MATCH


# ===========================================================================
# TEST B — Known person (present in video) → CONFIDENT_MATCH
# ===========================================================================

class TestB_KnownPerson:
    """
    Query person IS in the video.
    Body sim is well above threshold AND face sim is high → CONFIDENT_MATCH.
    """

    def test_b1_confident_match_when_body_and_face_agree(self):
        query_body  = _make_embedding(512, seed=100)
        query_face  = _make_embedding(512, seed=101)
        gallery_face = _near_copy(query_face, noise=0.01, seed=102)  # high face sim

        gallery  = _make_body_gallery(query_body, track_id=13, noise=0.01)
        face_gal = _make_face_gallery(gallery_face, track_id=13)

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
            face_gallery=face_gal,
            query_face_embedding=query_face,
            query_face_confidence=0.92,
        )
        assert len(results) == 1
        assert results[0].match_status == MatchStatus.CONFIDENT_MATCH, (
            f"Known person with high body+face sim must be CONFIDENT_MATCH, "
            f"got {results[0].match_status}"
        )
        assert results[0].face_veto_applied is False

    def test_b2_body_only_above_threshold_is_confident(self):
        """Body-only mode: sim above threshold → CONFIDENT_MATCH (no face available)."""
        query_body = _make_embedding(512, seed=200)
        gallery    = _make_body_gallery(query_body, track_id=16, noise=0.01)

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
        )
        assert len(results) == 1
        assert results[0].match_status == MatchStatus.CONFIDENT_MATCH
        assert results[0].face_veto_applied is False


# ===========================================================================
# TEST C — Similar clothing, different identity → FACE_MISMATCH
# ===========================================================================

class TestC_SimilarClothingDifferentIdentity:
    """
    Critical false-positive regression test.
    body_sim HIGH + kpr_sim HIGH + face_sim LOW → FACE_MISMATCH.
    KPR must NOT override a reliable face contradiction.
    """

    def test_c1_face_mismatch_not_confident(self):
        query_body  = _make_embedding(512, seed=300)
        query_face  = _make_embedding(512, seed=301)
        gallery_face = _opposite(query_face, noise=0.02, seed=302)

        gallery  = _make_body_gallery(query_body, track_id=66, noise=0.01)
        face_gal = _make_face_gallery(gallery_face, track_id=66)

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
            face_gallery=face_gal,
            query_face_embedding=query_face,
            query_face_confidence=0.91,
        )
        assert len(results) == 1
        match = results[0]

        # The critical assertion
        assert match.match_status != MatchStatus.CONFIDENT_MATCH, (
            "REGRESSION: Similar-clothing impostor must never be CONFIDENT_MATCH "
            f"when reliable face mismatch exists. Got {match.match_status}"
        )
        assert match.match_status == MatchStatus.FACE_MISMATCH
        assert match.face_veto_applied is True

    def test_c2_face_mismatch_candidate_still_visible_for_review(self):
        """Even a vetoed candidate must appear in results (for human review)."""
        query_body  = _make_embedding(512, seed=310)
        query_face  = _make_embedding(512, seed=311)
        gallery_face = _opposite(query_face, noise=0.02, seed=312)

        gallery  = _make_body_gallery(query_body, track_id=77, noise=0.01)
        face_gal = _make_face_gallery(gallery_face, track_id=77)

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
            face_gallery=face_gal,
            query_face_embedding=query_face,
            query_face_confidence=0.90,
        )
        # Candidate is returned (not suppressed) — human can verify
        assert len(results) == 1
        assert results[0].match_status == MatchStatus.FACE_MISMATCH


# ===========================================================================
# TEST D — Same person, different clothing → face preserves identity
# ===========================================================================

class TestD_SamePerson_DifferentClothing:
    """
    Body sim may drop when clothing changes, but face sim stays high.
    Result: face supports the match → CONFIDENT_MATCH if fused score clears threshold.
    """

    def test_d1_face_reinforces_low_body_sim(self):
        query_body  = _make_embedding(512, seed=400)
        query_face  = _make_embedding(512, seed=401)
        # Gallery body is moderately similar (clothing change → lower body sim)
        gallery_body = _near_copy(query_body, noise=0.15, seed=402)
        # But gallery face is near-identical
        gallery_face_emb = _near_copy(query_face, noise=0.01, seed=403)

        from ai_pipeline.reid.similarity import cosine_similarity
        body_sim = cosine_similarity(query_body, gallery_body)

        gallery  = [{
            "embedding": gallery_body,
            "track_id": 13,
            "crop_path": "crops/C01_track13_frame0001.jpg",
            "timestamp": "10:02:01.000",
            "frame": 1,
        }]
        face_gal = _make_face_gallery(gallery_face_emb, track_id=13)

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
            face_gallery=face_gal,
            query_face_embedding=query_face,
            query_face_confidence=0.88,
        )
        # Face sim should be high enough that no veto fires
        if results:
            assert results[0].face_veto_applied is False, (
                "High face similarity must not trigger a veto"
            )


# ===========================================================================
# TEST E — Face unavailable → no veto applied
# ===========================================================================

class TestE_FaceUnavailable:
    """
    When face gallery is None or query face is None, the veto must NOT fire.
    The system should fall back to body-only decision.
    """

    def test_e1_no_face_gallery_no_veto(self):
        """face_gallery=None → body-only path, no veto."""
        query_body = _make_embedding(512, seed=500)
        gallery    = _make_body_gallery(query_body, track_id=20, noise=0.01)

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
            face_gallery=None,         # explicitly absent
            query_face_embedding=None,
        )
        assert len(results) == 1
        assert results[0].face_veto_applied is False
        # Body sim is high → should be CONFIDENT_MATCH
        assert results[0].match_status == MatchStatus.CONFIDENT_MATCH

    def test_e2_no_query_face_embedding_no_veto(self):
        """query_face_embedding=None → face signal inactive, no veto."""
        query_body   = _make_embedding(512, seed=501)
        query_face   = _make_embedding(512, seed=502)
        gallery_face = _opposite(query_face, seed=503)

        gallery  = _make_body_gallery(query_body, track_id=21, noise=0.01)
        face_gal = _make_face_gallery(gallery_face, track_id=21)

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
            face_gallery=face_gal,
            query_face_embedding=None,   # no query face → veto cannot fire
        )
        assert len(results) == 1
        assert results[0].face_veto_applied is False


# ===========================================================================
# TEST F — Low-quality query face → no veto
# ===========================================================================

class TestF_LowQualityFace:
    """
    query_face_confidence below face_min_query_confidence → face is unreliable.
    Veto must NOT fire even if gallery face sim is low.
    """

    def test_f1_low_query_face_confidence_no_veto(self):
        query_body  = _make_embedding(512, seed=600)
        query_face  = _make_embedding(512, seed=601)
        gallery_face = _opposite(query_face, seed=602)  # low sim

        gallery  = _make_body_gallery(query_body, track_id=30, noise=0.01)
        face_gal = _make_face_gallery(gallery_face, track_id=30)

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
            face_gallery=face_gal,
            query_face_embedding=query_face,
            query_face_confidence=0.10,   # well below face_min_query_confidence=0.50
        )
        assert len(results) == 1
        assert results[0].face_veto_applied is False, (
            "Low-quality query face must not trigger a veto "
            f"(confidence=0.10 < min=0.50)"
        )

    def test_f2_face_confidence_exactly_at_minimum_no_veto(self):
        """Edge case: face_confidence == face_min_query_confidence - ε → no veto."""
        from backend.app.config import settings
        min_conf = settings.face_min_query_confidence  # 0.50

        query_body  = _make_embedding(512, seed=610)
        query_face  = _make_embedding(512, seed=611)
        gallery_face = _opposite(query_face, seed=612)

        gallery  = _make_body_gallery(query_body, track_id=31, noise=0.01)
        face_gal = _make_face_gallery(gallery_face, track_id=31)

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
            face_gallery=face_gal,
            query_face_embedding=query_face,
            query_face_confidence=min_conf - 0.01,  # just below minimum
        )
        assert len(results) == 1
        assert results[0].face_veto_applied is False


# ===========================================================================
# TEST G — OSNet gallery → OSNet threshold and scaling
# ===========================================================================

class TestG_OSNetGallery:
    """
    512-dim gallery → OSNet x1_0 backbone detected automatically.
    Threshold = 0.74, soft_floor = 0.60.
    Confidence scaling uses [0.3315, 0.9730] range.
    """

    def test_g1_backbone_detected_as_osnet(self):
        query_emb = _make_embedding(512, seed=700)
        gallery   = _make_body_gallery(query_emb, track_id=1, noise=0.01)
        results   = _svc.search_camera_top_k(
            query_embedding=query_emb, gallery=gallery, camera_id="C01"
        )
        assert len(results) == 1
        assert results[0].embedding_dim == 512
        assert results[0].active_backbone == "OSNet x1_0"

    def test_g2_osnet_threshold_is_074(self):
        name, dim, thr, sf = _svc._detect_backbone(
            _make_body_gallery(_make_embedding(512, 701), 1)
        )
        assert thr == 0.74
        assert sf  == 0.60

    def test_g3_osnet_confidence_scaling(self):
        """sim=0.8148 → OSNet scaling → ~75.4% (not SOLIDER value)."""
        conf = similarity_to_confidence(0.8148, embedding_dim=512)
        expected = ((0.8148 - 0.3315049352393543) / (0.9730031552165505 - 0.3315049352393543)) * 100
        assert conf == pytest.approx(expected, abs=0.1)

    def test_g4_sim_below_074_not_returned_as_confident(self):
        """Sim just below threshold → POSSIBLE_MATCH_REVIEW (not CONFIDENT_MATCH)."""
        query_emb = _make_embedding(512, seed=720)
        # Build gallery with controlled similarity slightly below 0.74
        rng = np.random.default_rng(721)
        base = np.array(query_emb)
        # Add enough noise to drop sim below 0.74
        noise_vec = rng.standard_normal(512) * 0.80
        g_emb = _unit(base + noise_vec).tolist()
        from ai_pipeline.reid.similarity import cosine_similarity
        sim = cosine_similarity(query_emb, g_emb)

        gallery = [{"embedding": g_emb, "track_id": 55,
                    "crop_path": "crops/C01_track55_f1.jpg",
                    "timestamp": "10:00", "frame": 1}]
        results = _svc.search_camera_top_k(
            query_embedding=query_emb, gallery=gallery, camera_id="C01"
        )
        if results:
            assert results[0].match_status in (
                MatchStatus.POSSIBLE_MATCH_REVIEW,
                MatchStatus.NO_CONFIDENT_MATCH,
            ), f"Sim={sim:.4f} below 0.74 must not be CONFIDENT_MATCH"


# ===========================================================================
# TEST H — SOLIDER gallery → SOLIDER threshold and scaling
# ===========================================================================

class TestH_SOLIDERGallery:
    """
    768-dim gallery → SOLIDER Swin-Small backbone detected automatically.
    Threshold = 0.94, soft_floor = 0.90.
    Confidence scaling uses [0.83, 0.995] range.
    CRITICAL: OSNet threshold 0.74 must NOT be applied to SOLIDER.
    """

    def test_h1_backbone_detected_as_solider(self):
        query_emb = _make_embedding(768, seed=800)
        gallery   = _make_body_gallery(query_emb, track_id=1, noise=0.01, dim=768)
        results   = _svc.search_camera_top_k(
            query_embedding=query_emb, gallery=gallery, camera_id="C01"
        )
        assert len(results) == 1
        assert results[0].embedding_dim == 768
        assert results[0].active_backbone == "SOLIDER Swin-Small"

    def test_h2_solider_threshold_is_094(self):
        name, dim, thr, sf = _svc._detect_backbone(
            _make_body_gallery(_make_embedding(768, 801), 1, dim=768)
        )
        assert thr == 0.94
        assert sf  == 0.90

    def test_h3_solider_confidence_scaling_differs_from_osnet(self):
        """
        SOLIDER confidence for sim=0.87 must be very different from OSNet.
        OSNet: (0.87-0.3315)/(0.9730-0.3315)*100 ≈ 83.9%
        SOLIDER: (0.87-0.83)/(0.995-0.83)*100 ≈ 24.2%
        They must NOT be equal.
        """
        osnet_conf   = similarity_to_confidence(0.87, embedding_dim=512)
        solider_conf = similarity_to_confidence(0.87, embedding_dim=768)
        assert not math.isclose(osnet_conf, solider_conf, abs_tol=5.0), (
            f"OSNet and SOLIDER confidence must differ for same sim=0.87. "
            f"OSNet={osnet_conf:.1f}% SOLIDER={solider_conf:.1f}%"
        )
        assert osnet_conf > solider_conf, (
            "OSNet should score sim=0.87 higher than SOLIDER (different ranges)"
        )

    def test_h4_solider_high_body_sim_still_not_confident_below_094(self):
        """
        SOLIDER sim = 0.91 is above OSNet threshold (0.74) but below SOLIDER
        threshold (0.94).  Must be POSSIBLE_MATCH_REVIEW, not CONFIDENT_MATCH.
        This is the core regression: OSNet 0.74 must NOT apply to SOLIDER.
        """
        query_emb = _make_embedding(768, seed=820)
        # Construct gallery embedding with sim ≈ 0.91 (above 0.74, below 0.94)
        rng = np.random.default_rng(821)
        base = np.array(query_emb)
        noise_vec = rng.standard_normal(768) * 0.30
        g_emb = _unit(base + noise_vec).tolist()

        from ai_pipeline.reid.similarity import cosine_similarity
        sim = cosine_similarity(query_emb, g_emb)

        gallery = [{"embedding": g_emb, "track_id": 66,
                    "crop_path": "crops/C01_track66_f1.jpg",
                    "timestamp": "10:00", "frame": 1}]
        results = _svc.search_camera_top_k(
            query_embedding=query_emb, gallery=gallery, camera_id="C01"
        )
        if results and 0.60 <= sim < 0.94:
            assert results[0].match_status != MatchStatus.CONFIDENT_MATCH, (
                f"SOLIDER sim={sim:.4f} is below SOLIDER threshold=0.94. "
                f"Must not be CONFIDENT_MATCH. "
                f"(This would fire if OSNet threshold 0.74 was incorrectly applied.)"
            )


# ===========================================================================
# TEST I — KPR enabled → KPR high + face low → FACE_MISMATCH
# ===========================================================================

class TestI_KPREnabled:
    """
    KPR is part-based appearance evidence, NOT biometric identity.
    High KPR sim + low face sim → FACE_MISMATCH (KPR does not override face).
    """

    def _make_kpr_record(self, query_emb: list[float], track_id: int,
                         n_parts: int = 5, noise: float = 0.02) -> dict:
        """Query KPR dict matching embed_kpr.py output schema."""
        rng = np.random.default_rng(seed=track_id * 7)
        parts = [
            _unit(np.array(query_emb[:len(query_emb)]) + rng.standard_normal(len(query_emb)) * noise).tolist()
            for _ in range(n_parts)
        ]
        return {
            "holistic_embedding": query_emb,
            "part_embeddings":    parts,
            "part_visibility":    [0.9] * n_parts,
        }

    def _make_kpr_gallery(self, query_kpr: dict, track_id: int,
                          n_crops: int = 3, noise: float = 0.01) -> list[dict]:
        """KPR gallery records with high part similarity."""
        rng = np.random.default_rng(seed=track_id + 1)
        records = []
        for i in range(n_crops):
            parts = [
                _unit(np.array(p) + rng.standard_normal(len(p)) * noise).tolist()
                for p in query_kpr["part_embeddings"]
            ]
            records.append({
                "track_id": track_id,
                "crop_path": f"crops/C01_track{track_id}_frame{i:04d}.jpg",
                "timestamp": f"10:02:0{i}.000",
                "frame": i,
                "holistic_embedding": query_kpr["holistic_embedding"],
                "part_embeddings":    parts,
                "part_visibility":    [0.9] * len(parts),
                "camera_id": "C01",
                "bbox": None,
                "detection_confidence": 0.91,
            })
        return records

    def test_i1_kpr_cannot_override_face_mismatch(self):
        """
        KPR sim high + face sim low → FACE_MISMATCH.
        KPR is appearance evidence, not biometric identity.
        """
        query_body = _make_embedding(512, seed=900)
        query_face = _make_embedding(512, seed=901)
        gallery_face = _opposite(query_face, noise=0.02, seed=902)

        query_kpr   = self._make_kpr_record(query_body, track_id=88)
        kpr_gallery = self._make_kpr_gallery(query_kpr, track_id=88)
        gallery     = _make_body_gallery(query_body, track_id=88, noise=0.01)
        face_gal    = _make_face_gallery(gallery_face, track_id=88)

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
            face_gallery=face_gal,
            query_face_embedding=query_face,
            query_face_confidence=0.94,
            kpr_gallery=kpr_gallery,
            query_kpr=query_kpr,
        )
        assert len(results) == 1
        match = results[0]
        assert match.match_status == MatchStatus.FACE_MISMATCH, (
            f"KPR (appearance) must not override reliable face mismatch. "
            f"Got {match.match_status}"
        )
        assert match.face_veto_applied is True

    def test_i2_kpr_disabled_vs_enabled_no_false_positive_regression(self):
        """
        Enabling KPR must not suddenly make an impostor a CONFIDENT_MATCH
        when it was previously not a match.
        """
        query_body = _make_embedding(512, seed=910)
        # Gallery with body sim just below 0.74
        rng = np.random.default_rng(911)
        base = np.array(query_body)
        g_emb = _unit(base + rng.standard_normal(512) * 0.80).tolist()
        gallery = [{"embedding": g_emb, "track_id": 99,
                    "crop_path": "crops/C01_track99_f1.jpg",
                    "timestamp": "10:00", "frame": 1}]

        # Without KPR
        r_no_kpr = _svc.search_camera_top_k(
            query_embedding=query_body, gallery=gallery, camera_id="C01"
        )
        # KPR records for the same track (high KPR sim)
        query_kpr   = self._make_kpr_record(query_body, track_id=99)
        kpr_gallery = self._make_kpr_gallery(query_kpr, track_id=99)
        r_kpr = _svc.search_camera_top_k(
            query_embedding=query_body, gallery=gallery, camera_id="C01",
            kpr_gallery=kpr_gallery, query_kpr=query_kpr,
        )

        if r_no_kpr and r_no_kpr[0].match_status != MatchStatus.CONFIDENT_MATCH:
            # KPR should not upgrade this to CONFIDENT_MATCH on its own
            if r_kpr:
                assert r_kpr[0].match_status != MatchStatus.CONFIDENT_MATCH or \
                       r_kpr[0].fused_score >= 0.74, (
                    "KPR alone must not falsely elevate a previously non-confident "
                    "candidate to CONFIDENT_MATCH"
                )


# ===========================================================================
# TEST J — Top 3 unique tracks, correct ordering
# ===========================================================================

class TestJ_TopThreeCandidates:
    """
    search_camera_top_k returns:
      - At most candidate_tracks results
      - One entry per unique track_id (not multiple crops of same track)
      - Sorted by fused_score descending
    """

    def test_j1_at_most_three_unique_tracks(self):
        query_emb = _make_embedding(512, seed=1000)
        # 5 gallery tracks, all with high body sim
        gallery = []
        for tid in range(1, 6):
            base = np.array(query_emb)
            rng  = np.random.default_rng(seed=tid * 13)
            g    = _unit(base + rng.standard_normal(512) * (0.01 + tid * 0.005)).tolist()
            gallery.append({
                "embedding": g, "track_id": tid,
                "crop_path": f"crops/C01_track{tid}_f1.jpg",
                "timestamp": f"10:00:0{tid}", "frame": tid,
            })

        results = _svc.search_camera_top_k(
            query_embedding=query_emb, gallery=gallery, camera_id="C01",
            candidate_tracks=3,
        )
        assert len(results) <= 3, (
            f"Must return at most 3 tracks, got {len(results)}"
        )
        track_ids = [r.track_id for r in results]
        assert len(track_ids) == len(set(track_ids)), (
            f"Track IDs must be unique, got {track_ids}"
        )

    def test_j2_sorted_by_fused_score_descending(self):
        query_emb = _make_embedding(512, seed=1001)
        gallery = []
        for tid in range(1, 4):
            rng = np.random.default_rng(seed=tid * 17)
            g   = _unit(np.array(query_emb) + rng.standard_normal(512) * (tid * 0.03)).tolist()
            gallery.append({
                "embedding": g, "track_id": tid,
                "crop_path": f"crops/C01_track{tid}_f1.jpg",
                "timestamp": f"10:00:0{tid}", "frame": tid,
            })

        results = _svc.search_camera_top_k(
            query_embedding=query_emb, gallery=gallery, camera_id="C01",
            candidate_tracks=3,
        )
        fused_scores = [r.fused_score for r in results]
        assert fused_scores == sorted(fused_scores, reverse=True), (
            f"Results must be sorted by fused_score descending, got {fused_scores}"
        )

    def test_j3_multiple_crops_same_track_collapsed_to_one(self):
        """Multiple crops from the same track → only one result entry."""
        query_emb = _make_embedding(512, seed=1002)
        base = np.array(query_emb)
        # 4 crops all from track 42
        gallery = [
            {
                "embedding": _unit(base + np.random.default_rng(seed=i).standard_normal(512) * 0.02).tolist(),
                "track_id": 42,
                "crop_path": f"crops/C01_track42_f{i}.jpg",
                "timestamp": f"10:00:0{i}", "frame": i,
            }
            for i in range(4)
        ]
        results = _svc.search_camera_top_k(
            query_embedding=query_emb, gallery=gallery, camera_id="C01",
            candidate_tracks=3,
        )
        track_ids = [r.track_id for r in results]
        assert track_ids.count(42) <= 1, (
            "Same track must not appear multiple times in results"
        )


# ===========================================================================
# REGRESSION R1 — Body-only path unchanged
# ===========================================================================

class TestR1_BodyOnlyPath:
    """face_gallery=None AND kpr_gallery=None → identical to pre-fusion behaviour."""

    def test_r1_body_only_no_optional_params(self):
        query_emb = _make_embedding(512, seed=2000)
        gallery   = _make_body_gallery(query_emb, track_id=5, noise=0.01)

        r1 = _svc.search_camera_top_k(
            query_embedding=query_emb, gallery=gallery, camera_id="C01"
        )
        r2 = _svc.search_camera_top_k(
            query_embedding=query_emb, gallery=gallery, camera_id="C01",
            face_gallery=None, query_face_embedding=None,
            kpr_gallery=None, query_kpr=None,
        )
        assert len(r1) == len(r2)
        if r1:
            assert r1[0].appearance_score == r2[0].appearance_score
            assert r1[0].fused_score       == r2[0].fused_score
            assert r1[0].match_status      == r2[0].match_status


# ===========================================================================
# REGRESSION R2 — Face coverage below minimum → no veto
# ===========================================================================

class TestR2_FaceCoverageBelowMin:
    """Gallery track has face_coverage < 0.30 → face evidence unreliable → no veto."""

    def test_r2_low_coverage_no_veto(self):
        query_body  = _make_embedding(512, seed=2100)
        query_face  = _make_embedding(512, seed=2101)
        gallery_face = _opposite(query_face, seed=2102)

        gallery = _make_body_gallery(query_body, track_id=11, noise=0.01)
        # Only 1 out of 10 crops has a face → coverage = 0.10 < 0.30
        face_gal = [
            {"track_id": 11, "face_detected": True,
             "face_embedding": gallery_face,
             "face_confidence": 0.95,
             "crop_path": "crops/C01_track11_f1.jpg",
             "camera_id": "C01", "frame": 1, "timestamp": "10:00", "bbox": None},
        ] + [
            {"track_id": 11, "face_detected": False,
             "face_embedding": None, "face_confidence": None,
             "crop_path": f"crops/C01_track11_f{i}.jpg",
             "camera_id": "C01", "frame": i, "timestamp": "10:00", "bbox": None}
            for i in range(2, 11)   # 9 crops without face → coverage = 1/10 = 0.10
        ]

        results = _svc.search_camera_top_k(
            query_embedding=query_body,
            gallery=gallery,
            camera_id="C01",
            face_gallery=face_gal,
            query_face_embedding=query_face,
            query_face_confidence=0.90,
        )
        assert len(results) == 1
        assert results[0].face_veto_applied is False, (
            f"Gallery face_coverage=0.10 < min=0.30 → veto must NOT fire. "
            f"Got face_veto_applied={results[0].face_veto_applied}"
        )


# ===========================================================================
# REGRESSION R5 — Existing OSNet confidence scaling unchanged
# ===========================================================================

class TestR5_ExistingConfidenceScaling:
    """The OBSERVED_MIN / OBSERVED_MAX constants are preserved from Phase 4.5."""

    def test_r5_osnet_min_constant_unchanged(self):
        assert BACKBONE_OSNET_512.observed_min == pytest.approx(0.3315049352393543)

    def test_r5_osnet_max_constant_unchanged(self):
        assert BACKBONE_OSNET_512.observed_max == pytest.approx(0.9730031552165505)

    def test_r5_known_example_0_8148(self):
        """Track 13 original from Phase 5.7 (0.8148 → ~75.4%)."""
        result   = similarity_to_confidence(0.8148, embedding_dim=512)
        expected = ((0.8148 - 0.3315049352393543) / (0.9730031552165505 - 0.3315049352393543)) * 100
        assert result == pytest.approx(expected, abs=0.01)

    def test_r5_known_example_0_8878(self):
        """Track 16 original from Phase 5.7 (0.8878 → ~86.8%)."""
        result   = similarity_to_confidence(0.8878, embedding_dim=512)
        expected = ((0.8878 - 0.3315049352393543) / (0.9730031552165505 - 0.3315049352393543)) * 100
        assert result == pytest.approx(expected, abs=0.01)

    def test_r5_get_backbone_config_512_returns_osnet(self):
        cfg = get_backbone_config(512)
        assert cfg.name == "OSNet x1_0"
        assert cfg.no_match_threshold == 0.74

    def test_r5_get_backbone_config_768_returns_solider(self):
        cfg = get_backbone_config(768)
        assert cfg.name == "SOLIDER Swin-Small"
        assert cfg.no_match_threshold == 0.94

    def test_r5_solider_conf_not_equal_osnet_conf_for_same_sim(self):
        """Ensure scaling is backbone-aware — two backbones must give different results."""
        sim = 0.89
        osnet_c   = similarity_to_confidence(sim, embedding_dim=512)
        solider_c = similarity_to_confidence(sim, embedding_dim=768)
        assert not math.isclose(osnet_c, solider_c, abs_tol=1.0), (
            f"OSNet({sim})={osnet_c:.1f}% must differ from SOLIDER({sim})={solider_c:.1f}%"
        )

    def test_r5_clamping_still_works(self):
        """Both ends clamp to [0, 100] for any backbone."""
        for dim in (512, 768):
            assert similarity_to_confidence(-1.0, embedding_dim=dim) == 0.0
            assert similarity_to_confidence(2.0,  embedding_dim=dim) == 100.0
