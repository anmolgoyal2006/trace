"""
test_kpr_evidence_matching.py — XAI Phase 3
Tests that MatchingService attaches per-part KPR evidence to CameraMatch
WITHOUT changing any matching decision.

These tests:
  - Do NOT require GPU, torchreid, or the KPR model.
  - Use synthetic embeddings with exactly controlled cosine similarities.

Run:
    pytest ai_pipeline/reid/test_kpr_evidence_matching.py -v

Coverage:
  EV1  No-regression — ranking, fused_score, match_status, kpr_sim and
       mean_visible_parts are identical to the evidence-free code path
  EV2  Evidence presence — None without KPR, populated with KPR
  EV3  Evidence content — belongs to the candidate's own track and
       agrees with kpr_sim / mean_visible_parts
  EV4  Edge cases — unusable KPR track, face veto still applied
  EV5  No raw kpr_evidence field on API schemas — evidence reaches the API
       only inside CandidateTrackOut.explanation (see test_candidate_explanation_api)
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

import backend.app.services.matching_service as ms_module
from ai_pipeline.reid.kpr_similarity import aggregate_kpr_by_track
from backend.app.services.matching_service import MatchingService, MatchStatus

_svc = MatchingService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_emb(seed: int, dim: int = 512) -> list[float]:
    v = np.random.default_rng(seed).standard_normal(dim)
    return (v / np.linalg.norm(v)).tolist()


def _emb_with_sim(query: list[float], target_sim: float, seed: int = 99) -> list[float]:
    """Unit vector g with cosine_similarity(query, g) == target_sim (to ~1e-16)."""
    q = np.array(query)
    q_hat = q / np.linalg.norm(q)
    r = np.random.default_rng(seed).standard_normal(len(q))
    r -= (r @ q_hat) * q_hat
    r /= np.linalg.norm(r)
    g = target_sim * q_hat + math.sqrt(max(0.0, 1.0 - target_sim ** 2)) * r
    return (g / np.linalg.norm(g)).tolist()


def _query_kpr(base: list[float], n_parts: int = 5) -> dict:
    parts = [_emb_with_sim(base, 0.99, seed=i + 200) for i in range(n_parts)]
    return {"holistic_embedding": base,
            "part_embeddings": parts, "part_visibility": [0.95] * n_parts}


def _scenario(seed: int = 1):
    """
    Three tracks. Body and KPR similarities are chosen so KPR re-orders
    the body ranking — any decision drift would show up in EV1.

        track   body   KPR part sims (per crop)           visibility
        10      0.82   0.70, 0.75                          all visible
        20      0.78   0.95, 0.60  (crop 0 is best)        leg parts occluded
        30      0.75   0.88                                part 1 below threshold
    """
    q = _base_emb(seed)
    body = {10: 0.82, 20: 0.78, 30: 0.75}
    gallery = [
        {"embedding": _emb_with_sim(q, s, seed=tid), "track_id": tid,
         "crop_path": f"crops/t{tid}_f0.jpg", "timestamp": "10:00:01.000", "frame": 1}
        for tid, s in body.items()
    ]

    q_kpr = _query_kpr(q)
    crops = {
        10: [(0.70, [0.95] * 5), (0.75, [0.95] * 5)],
        20: [(0.95, [0.9, 0.9, 0.9, 0.1, 0.1]), (0.60, [0.95] * 5)],
        30: [(0.88, [0.9, 0.2, 0.9, 0.9, 0.9])],
    }
    kpr_gallery = []
    for tid, specs in crops.items():
        for ci, (part_sim, vis) in enumerate(specs):
            parts = [_emb_with_sim(p, part_sim, seed=tid * 100 + ci * 10 + i)
                     for i, p in enumerate(q_kpr["part_embeddings"])]
            kpr_gallery.append({
                "track_id": tid, "crop_path": f"crops/kpr_t{tid}_c{ci}.jpg",
                "timestamp": f"10:00:0{ci}.000", "frame": ci,
                "holistic_embedding": q, "part_embeddings": parts,
                "part_visibility": vis, "camera_id": "C01",
            })
    return q, gallery, q_kpr, kpr_gallery


def _search(q, gallery, kpr_gallery=None, q_kpr=None, **kw):
    return _svc.search_camera_top_k(
        query_embedding=q, gallery=gallery, camera_id="C01",
        kpr_gallery=kpr_gallery, query_kpr=q_kpr, **kw,
    )


def _decision_view(matches) -> list[tuple]:
    return [(m.track_id, m.fused_score, m.match_status, m.kpr_sim,
             m.mean_visible_parts, m.best_confidence, m.face_veto_applied)
            for m in matches]


@pytest.fixture
def evidence_disabled(monkeypatch):
    """Force the pre-XAI code path: aggregate_kpr_by_track without evidence."""
    def _no_evidence(*args, **kwargs):
        kwargs["include_evidence"] = False
        return aggregate_kpr_by_track(*args, **kwargs)
    monkeypatch.setattr(ms_module, "aggregate_kpr_by_track", _no_evidence)


# ===========================================================================
# EV1 — No regression in matching decisions
# ===========================================================================

class TestEV1_NoDecisionChange:

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_decisions_identical_to_evidence_free_path(self, seed, monkeypatch):
        q, gallery, q_kpr, kpr_gallery = _scenario(seed)
        with_ev = _search(q, gallery, kpr_gallery, q_kpr)

        def _no_evidence(*args, **kwargs):
            kwargs["include_evidence"] = False
            return aggregate_kpr_by_track(*args, **kwargs)
        monkeypatch.setattr(ms_module, "aggregate_kpr_by_track", _no_evidence)
        without_ev = _search(q, gallery, kpr_gallery, q_kpr)

        assert with_ev, "scenario must produce candidates"
        assert _decision_view(with_ev) == _decision_view(without_ev)

    def test_evidence_disabled_path_has_no_evidence(self, evidence_disabled):
        q, gallery, q_kpr, kpr_gallery = _scenario()
        for m in _search(q, gallery, kpr_gallery, q_kpr):
            assert m.kpr_evidence is None

    def test_kpr_reorders_body_ranking(self):
        # Sanity: the scenario actually exercises KPR-driven ranking
        q, gallery, q_kpr, kpr_gallery = _scenario()
        body_order = [m.track_id for m in _search(q, gallery)]
        kpr_order  = [m.track_id for m in _search(q, gallery, kpr_gallery, q_kpr)]
        assert body_order == [10, 20, 30]
        assert kpr_order != body_order


# ===========================================================================
# EV2 — Evidence presence
# ===========================================================================

class TestEV2_Presence:

    def test_body_only_has_no_evidence(self):
        q, gallery, _, _ = _scenario()
        for m in _search(q, gallery):
            assert m.kpr_evidence is None

    def test_kpr_gallery_without_query_has_no_evidence(self):
        q, gallery, _, kpr_gallery = _scenario()
        for m in _search(q, gallery, kpr_gallery, None):
            assert m.kpr_evidence is None

    def test_every_kpr_candidate_has_evidence(self):
        q, gallery, q_kpr, kpr_gallery = _scenario()
        res = _search(q, gallery, kpr_gallery, q_kpr)
        assert len(res) == 3
        for m in res:
            assert isinstance(m.kpr_evidence, dict)


# ===========================================================================
# EV3 — Evidence content
# ===========================================================================

class TestEV3_Content:

    def _by_track(self):
        q, gallery, q_kpr, kpr_gallery = _scenario()
        return {m.track_id: m for m in _search(q, gallery, kpr_gallery, q_kpr)}

    def test_evidence_score_matches_kpr_sim(self):
        for m in self._by_track().values():
            assert m.kpr_evidence["score"] == pytest.approx(m.kpr_sim, abs=1e-6)

    def test_evidence_belongs_to_own_track_best_crop(self):
        by = self._by_track()
        assert by[10].kpr_evidence["crop_path"] == "crops/kpr_t10_c1.jpg"
        assert by[20].kpr_evidence["crop_path"] == "crops/kpr_t20_c0.jpg"
        assert by[30].kpr_evidence["crop_path"] == "crops/kpr_t30_c0.jpg"

    def test_part_statuses_reflect_visibility(self):
        by = self._by_track()
        assert [p["status"] for p in by[10].kpr_evidence["parts"]] == ["used"] * 5
        assert [p["status"] for p in by[20].kpr_evidence["parts"]] == \
            ["used", "used", "used", "below_threshold", "below_threshold"]
        assert [p["status"] for p in by[30].kpr_evidence["parts"]] == \
            ["used", "below_threshold", "used", "used", "used"]
        assert by[20].kpr_evidence["num_mutually_visible"] == 3

    def test_contributions_sum_to_score(self):
        for m in self._by_track().values():
            ev = m.kpr_evidence
            assert ev["fallback"] == "none"
            assert sum(p["contribution"] for p in ev["parts"]) == \
                pytest.approx(ev["score"], abs=1e-12)

    def test_single_crop_track_visible_parts_agree(self):
        # mean_visible_parts is a track average; for a 1-crop track it must
        # equal the evidence crop's mutual-visibility count
        m = self._by_track()[30]
        assert m.mean_visible_parts == m.kpr_evidence["num_mutually_visible"] == 4

    def test_evidence_json_serialisable(self):
        for m in self._by_track().values():
            assert json.loads(json.dumps(m.kpr_evidence)) == m.kpr_evidence


# ===========================================================================
# EV4 — Edge cases
# ===========================================================================

class TestEV4_EdgeCases:

    def test_track_missing_from_kpr_gallery_has_no_evidence(self):
        q, gallery, q_kpr, kpr_gallery = _scenario()
        kpr_gallery = [r for r in kpr_gallery if r["track_id"] != 30]
        by = {m.track_id: m for m in _search(q, gallery, kpr_gallery, q_kpr)}
        assert by[30].kpr_sim is None
        assert by[30].kpr_evidence is None

    def test_all_unusable_kpr_track_has_none_evidence(self):
        q, gallery, q_kpr, kpr_gallery = _scenario()
        for r in kpr_gallery:
            if r["track_id"] == 30:
                r["part_embeddings"] = r["part_embeddings"][:3]   # part-count mismatch
                r["part_visibility"] = r["part_visibility"][:3]
                r.pop("holistic_embedding")                       # no fallback
        # kpr_sim=0.0 drags track 30 below soft_floor, so check the helper directly
        info = _svc._compute_kpr_similarities(q_kpr, kpr_gallery)
        assert info[30]["kpr_sim"] == 0.0
        assert info[30]["kpr_evidence"] is None
        assert info[10]["kpr_evidence"] is not None

    def test_face_veto_still_applies_with_evidence(self):
        q, gallery, q_kpr, kpr_gallery = _scenario()
        q_face = _base_emb(500)
        face_gallery = [
            {"track_id": 10, "face_detected": True,
             "face_embedding": _emb_with_sim(q_face, -0.10, seed=600 + i),
             "face_confidence": 0.92, "crop_path": f"crops/t10_f{i}.jpg",
             "camera_id": "C01", "frame": i, "timestamp": f"10:00:0{i}", "bbox": None}
            for i in range(3)
        ]
        res = _search(q, gallery, kpr_gallery, q_kpr,
                      face_gallery=face_gallery, query_face_embedding=q_face,
                      query_face_confidence=0.95)
        m10 = next(m for m in res if m.track_id == 10)
        assert m10.match_status == MatchStatus.FACE_MISMATCH
        assert m10.kpr_evidence is not None


# ===========================================================================
# EV5 — No raw kpr_evidence field on API schemas
# ===========================================================================

class TestEV5_NotExposed:

    def test_api_schemas_have_no_evidence_field(self):
        try:
            from backend.app.models.schemas import CandidateTrackOut, SightingOut
        except TypeError as exc:   # schemas.py uses PEP 604 unions (Python >= 3.10)
            pytest.skip(f"backend schemas need Python >= 3.10: {exc}")
        assert "kpr_evidence" not in SightingOut.model_fields
        assert "kpr_evidence" not in CandidateTrackOut.model_fields
