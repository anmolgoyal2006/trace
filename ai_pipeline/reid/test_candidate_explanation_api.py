"""
test_candidate_explanation_api.py — XAI Phase 5
Tests that top candidates returned by the API carry the XAI explanation
(fusion breakdown + KPR part evidence) without changing existing fields.

Path under test:
    MatchingService.search_camera_top_k → CameraMatch
      → pipeline_service._build_top_candidates → CandidateTrackOut
      → session.top_candidates_json (json.dumps of model_dump)
      → GET /api/queries/{id}/route  (CandidateTrackOut(**json.loads(...)))

These tests:
  - Do NOT require GPU, torchreid, a database, or a running server.
  - Need Python >= 3.10 (backend schemas / pipeline_service); the module is
    skipped on older interpreters.

Run:
    pytest ai_pipeline/reid/test_candidate_explanation_api.py -v

Coverage:
  API1  _build_top_candidates maps fusion_explanation + kpr_evidence
  API2  Existing candidate fields are unchanged by the new field
  API3  JSON round-trip identical to the route endpoint's storage path
  API4  Legacy stored sessions (no "explanation" key) still parse
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

try:
    from backend.app.models.schemas import CandidateTrackOut
    from backend.app.services.pipeline_service import _build_top_candidates
except (SyntaxError, TypeError) as exc:   # backend needs Python >= 3.10
    pytest.skip(f"backend modules need Python >= 3.10: {exc}", allow_module_level=True)

from backend.app.services.matching_service import CameraMatch, MatchingService, MatchStatus

_svc = MatchingService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_emb(seed: int, dim: int = 512) -> list[float]:
    v = np.random.default_rng(seed).standard_normal(dim)
    return (v / np.linalg.norm(v)).tolist()


def _emb_with_sim(query: list[float], target_sim: float, seed: int = 99) -> list[float]:
    q = np.array(query)
    q_hat = q / np.linalg.norm(q)
    r = np.random.default_rng(seed).standard_normal(len(q))
    r -= (r @ q_hat) * q_hat
    r /= np.linalg.norm(r)
    g = target_sim * q_hat + math.sqrt(max(0.0, 1.0 - target_sim ** 2)) * r
    return (g / np.linalg.norm(g)).tolist()


def _camera_candidates() -> dict[str, list[CameraMatch]]:
    """Two cameras; C01 has KPR evidence, C02 is body-only."""
    out: dict[str, list[CameraMatch]] = {}
    q = _base_emb(1)
    q_kpr = {"holistic_embedding": q,
             "part_embeddings": [_emb_with_sim(q, 0.99, seed=200 + i) for i in range(5)],
             "part_visibility": [0.95, 0.95, 0.95, 0.1, 0.1]}

    gallery_c01 = [{"embedding": _emb_with_sim(q, s, seed=tid), "track_id": tid,
                    "crop_path": f"crops/C01_t{tid}.jpg", "timestamp": "10:00:01.000", "frame": 1}
                   for tid, s in [(1, 0.85), (2, 0.75)]]
    kpr_c01 = [{"track_id": tid, "crop_path": f"crops/C01_kpr{tid}.jpg",
                "holistic_embedding": q, "part_visibility": [0.9] * 5,
                "part_embeddings": [_emb_with_sim(p, ps, seed=300 + tid * 10 + i)
                                    for i, p in enumerate(q_kpr["part_embeddings"])]}
               for tid, ps in [(1, 0.92), (2, 0.70)]]
    out["C01"] = _svc.search_camera_top_k(
        query_embedding=q, gallery=gallery_c01, camera_id="C01",
        kpr_gallery=kpr_c01, query_kpr=q_kpr,
    )

    gallery_c02 = [{"embedding": _emb_with_sim(q, 0.80, seed=7), "track_id": 7,
                    "crop_path": "crops/C02_t7.jpg", "timestamp": "10:05:00.000", "frame": 1}]
    out["C02"] = _svc.search_camera_top_k(query_embedding=q, gallery=gallery_c02, camera_id="C02")
    return out


def _by_key(cands) -> dict:
    return {(c.camera_id, c.track_id): c for c in cands}


# ===========================================================================
# API1 — Mapping
# ===========================================================================

class TestAPI1_Mapping:

    def test_explanation_mapped_from_camera_match(self):
        cams = _camera_candidates()
        matches = {(m.camera_id, m.track_id): m for ms in cams.values() for m in ms}
        cands = _build_top_candidates(cams, n=3)
        assert len(cands) == 3
        for c in cands:
            m = matches[(c.camera_id, c.track_id)]
            assert c.explanation == {
                "fusion":       m.fusion_explanation,
                "kpr_evidence": m.kpr_evidence,
            }

    def test_kpr_and_body_only_candidates(self):
        by = _by_key(_build_top_candidates(_camera_candidates(), n=3))
        kpr_c = by[("C01", 1)]
        assert kpr_c.explanation["kpr_evidence"]["crop_path"] == "crops/C01_kpr1.jpg"
        assert [p["status"] for p in kpr_c.explanation["kpr_evidence"]["parts"]] == \
            ["used", "used", "used", "below_threshold", "below_threshold"]
        assert kpr_c.explanation["fusion"]["mode"] == "weighted"

        body_c = by[("C02", 7)]
        assert body_c.explanation["kpr_evidence"] is None
        assert body_c.explanation["fusion"]["mode"] == "body_only"

    def test_no_explanation_gives_none(self):
        m = CameraMatch(camera_id="C09", track_id=1, appearance_score=0.8,
                        mean_top3_similarity=0.8, best_confidence=70.0, mean_confidence=70.0,
                        fused_score=0.8, match_status=MatchStatus.CONFIDENT_MATCH)
        [c] = _build_top_candidates({"C09": [m]}, n=3)
        assert c.explanation is None


# ===========================================================================
# API2 — Existing fields unchanged
# ===========================================================================

class TestAPI2_ExistingFields:

    def test_legacy_fields_identical(self):
        cams = _camera_candidates()
        for c in _build_top_candidates(cams, n=3):
            m = next(m for m in cams[c.camera_id] if m.track_id == c.track_id)
            assert (c.best_confidence, c.appearance_score, c.fused_score, c.match_status,
                    c.face_veto_applied, c.face_sim, c.face_coverage, c.kpr_sim,
                    c.first_seen, c.crop_path, c.active_backbone) == \
                   (m.best_confidence, m.appearance_score, m.fused_score, m.match_status.value,
                    m.face_veto_applied, m.face_sim, m.face_coverage, m.kpr_sim,
                    m.first_seen, m.best_crop_path, m.active_backbone)

    def test_ordering_by_fused_score(self):
        cands = _build_top_candidates(_camera_candidates(), n=3)
        scores = [c.fused_score for c in cands]
        assert scores == sorted(scores, reverse=True)


# ===========================================================================
# API3 — JSON round-trip (same path as session.top_candidates_json)
# ===========================================================================

class TestAPI3_RoundTrip:

    def test_storage_round_trip_is_lossless(self):
        cands = _build_top_candidates(_camera_candidates(), n=3)
        stored = json.dumps([c.model_dump() for c in cands])          # pipeline_service
        restored = [CandidateTrackOut(**c) for c in json.loads(stored)]  # routers/queries.py
        assert restored == cands


# ===========================================================================
# API4 — Legacy stored sessions
# ===========================================================================

class TestAPI4_Legacy:

    def test_legacy_json_without_explanation_parses(self):
        legacy = {
            "camera_id": "C01", "track_id": 3, "best_confidence": 71.2,
            "appearance_score": 0.81, "fused_score": 0.79,
            "match_status": "CONFIDENT_MATCH", "face_veto_applied": False,
            "face_sim": None, "face_coverage": None, "kpr_sim": 0.77,
            "first_seen": "10:00:01.000", "crop_path": "crops/x.jpg",
            "active_backbone": "OSNet x1_0",
        }
        c = CandidateTrackOut(**json.loads(json.dumps(legacy)))
        assert c.explanation is None
        assert c.kpr_sim == 0.77
