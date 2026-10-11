"""
test_fusion_explanation.py — XAI Phase 4
Tests for CameraMatch.fusion_explanation: how fused_score and match_status
were reached (per-signal weights and contributions, excluded signals,
thresholds, decision reason).

These tests:
  - Do NOT require GPU, torchreid, or any model weights.
  - Use synthetic embeddings with exactly controlled cosine similarities.
  - Rely on test_fused_score_regression.py / test_matching_service.py for the
    guarantee that scores and decisions themselves are unchanged.

Run:
    pytest ai_pipeline/reid/test_fusion_explanation.py -v

Coverage:
  FX1  Body-only mode
  FX2  Weighted mode — weights, normalized weights, contributions
  FX3  Excluded signals and their reasons
  FX4  Decision reason, thresholds and margin agree with match_status
  FX5  Consistency invariants across many random scenarios
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

from backend.app.config import settings
from backend.app.services.matching_service import MatchingService, MatchStatus

_BW = settings.fusion_body_weight
_FW = settings.fusion_face_weight
_KW = settings.fusion_kpr_weight

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


def _gallery(q, body_sim: float, track_id: int = 1) -> list[dict]:
    return [{"embedding": _emb_with_sim(q, body_sim, seed=track_id), "track_id": track_id,
             "crop_path": f"crops/t{track_id}_f0.jpg", "timestamp": "10:00:01.000", "frame": 1}]


def _query_kpr(q) -> dict:
    parts = [_emb_with_sim(q, 0.99, seed=200 + i) for i in range(5)]
    return {"holistic_embedding": q, "part_embeddings": parts, "part_visibility": [0.95] * 5}


def _kpr_gallery(q_kpr: dict, part_sim: float, track_id: int = 1) -> list[dict]:
    parts = [_emb_with_sim(p, part_sim, seed=300 + i)
             for i, p in enumerate(q_kpr["part_embeddings"])]
    return [{"track_id": track_id, "crop_path": f"crops/kpr{track_id}.jpg",
             "holistic_embedding": q_kpr["holistic_embedding"],
             "part_embeddings": parts, "part_visibility": [0.95] * 5}]


def _face_gallery(q_face, face_sim: float, track_id: int = 1, detected: bool = True) -> list[dict]:
    emb = _emb_with_sim(q_face, face_sim, seed=400)
    return [{"track_id": track_id, "face_detected": detected,
             "face_embedding": emb if detected else None, "face_confidence": 0.92,
             "crop_path": f"crops/t{track_id}_f{i}.jpg"} for i in range(3)]


def _one(q, gallery, **kw):
    res = _svc.search_camera_top_k(query_embedding=q, gallery=gallery, camera_id="C01", **kw)
    assert res, "scenario must produce a candidate above soft_floor"
    return res[0]


def _signals(m) -> dict:
    return {s["signal"]: s for s in m.fusion_explanation["signals"]}


def _excluded(m) -> dict:
    return {e["signal"]: e["reason"] for e in m.fusion_explanation["excluded_signals"]}


# ===========================================================================
# FX1 — Body-only mode
# ===========================================================================

class TestFX1_BodyOnly:

    def test_single_body_signal_carries_full_weight(self):
        q = _base_emb(1)
        m = _one(q, _gallery(q, 0.90))
        ex = m.fusion_explanation
        assert ex["mode"] == "body_only"
        assert list(_signals(m)) == ["body"]
        body = _signals(m)["body"]
        assert body["normalized_weight"] == 1.0
        assert body["contribution"] == pytest.approx(m.appearance_score, abs=1e-6)
        assert ex["fused_score"] == body["similarity"]

    def test_excluded_signals_not_provided(self):
        q = _base_emb(2)
        m = _one(q, _gallery(q, 0.90))
        assert _excluded(m) == {
            "kpr":  "KPR signal not provided",
            "face": "face signal not provided",
        }


# ===========================================================================
# FX2 — Weighted mode
# ===========================================================================

class TestFX2_Weighted:

    def test_body_kpr_weights_and_contributions(self):
        q = _base_emb(3)
        qk = _query_kpr(q)
        m = _one(q, _gallery(q, 0.80), kpr_gallery=_kpr_gallery(qk, 0.90), query_kpr=qk)
        ex, sig = m.fusion_explanation, _signals(m)
        assert ex["mode"] == "weighted"
        assert list(sig) == ["body", "kpr"]
        assert sig["body"]["weight"] == _BW
        assert sig["kpr"]["weight"] == _KW
        assert sig["body"]["normalized_weight"] == pytest.approx(_BW / (_BW + _KW))
        assert sig["kpr"]["normalized_weight"] == pytest.approx(_KW / (_BW + _KW))
        assert sig["kpr"]["similarity"] == pytest.approx(m.kpr_sim, abs=1e-6)
        assert sum(s["contribution"] for s in sig.values()) == pytest.approx(ex["fused_score"], abs=1e-12)

    def test_all_three_signals(self):
        q, q_face = _base_emb(4), _base_emb(40)
        qk = _query_kpr(q)
        m = _one(q, _gallery(q, 0.80),
                 kpr_gallery=_kpr_gallery(qk, 0.90), query_kpr=qk,
                 face_gallery=_face_gallery(q_face, 0.70), query_face_embedding=q_face,
                 query_face_confidence=0.95)
        sig = _signals(m)
        assert list(sig) == ["body", "kpr", "face"]
        assert sig["face"]["weight"] == _FW
        assert sig["face"]["similarity"] == pytest.approx(m.face_sim, abs=1e-6)
        assert sum(s["normalized_weight"] for s in sig.values()) == pytest.approx(1.0)
        assert _excluded(m) == {}

    def test_fused_score_matches_match_field(self):
        q = _base_emb(5)
        qk = _query_kpr(q)
        m = _one(q, _gallery(q, 0.80), kpr_gallery=_kpr_gallery(qk, 0.70), query_kpr=qk)
        assert round(m.fusion_explanation["fused_score"], 6) == m.fused_score


# ===========================================================================
# FX3 — Excluded signals
# ===========================================================================

class TestFX3_Excluded:

    def test_kpr_provided_but_track_missing(self):
        q = _base_emb(6)
        qk = _query_kpr(q)
        m = _one(q, _gallery(q, 0.90, track_id=1),
                 kpr_gallery=_kpr_gallery(qk, 0.90, track_id=99), query_kpr=qk)
        assert "kpr" not in _signals(m)
        assert _excluded(m)["kpr"] == "no usable KPR record for this track"
        assert m.fusion_explanation["mode"] == "weighted"

    def test_negative_face_excluded_and_veto_explained(self):
        q, q_face = _base_emb(7), _base_emb(70)
        m = _one(q, _gallery(q, 0.90),
                 face_gallery=_face_gallery(q_face, -0.20), query_face_embedding=q_face,
                 query_face_confidence=0.95)
        assert m.match_status == MatchStatus.FACE_MISMATCH
        assert "face" not in _signals(m)
        assert "non-positive face similarity" in _excluded(m)["face"]
        ex = m.fusion_explanation
        assert ex["face_veto_applied"] is True
        assert ex["decision_reason"] == f"face veto: {m.face_veto_reason}"

    def test_no_gallery_face(self):
        q, q_face = _base_emb(8), _base_emb(80)
        m = _one(q, _gallery(q, 0.90),
                 face_gallery=_face_gallery(q_face, 0.5, detected=False),
                 query_face_embedding=q_face, query_face_confidence=0.95)
        assert _excluded(m)["face"] == "no gallery face embedding meeting coverage threshold"


# ===========================================================================
# FX4 — Decision reason, thresholds, margin
# ===========================================================================

class TestFX4_Decision:

    def test_confident_match_reason_and_margin(self):
        q = _base_emb(9)
        m = _one(q, _gallery(q, 0.90))
        ex = m.fusion_explanation
        assert m.match_status == MatchStatus.CONFIDENT_MATCH
        assert ex["margin_to_threshold"] >= 0
        assert ">= no_match_threshold" in ex["decision_reason"]
        assert ex["face_veto_applied"] is False

    def test_review_reason_and_margin(self):
        q = _base_emb(10)
        m = _one(q, _gallery(q, 0.68))
        ex = m.fusion_explanation
        assert m.match_status == MatchStatus.POSSIBLE_MATCH_REVIEW
        assert ex["margin_to_threshold"] < 0
        assert ex["soft_floor"] <= ex["fused_score"] < ex["no_match_threshold"]
        assert "human review" in ex["decision_reason"]

    def test_thresholds_follow_backbone(self):
        q512, q768 = _base_emb(11), _base_emb(11, dim=768)
        m512 = _one(q512, _gallery(q512, 0.95))
        m768 = _one(q768, _gallery(q768, 0.97))
        assert m512.fusion_explanation["no_match_threshold"] != \
            m768.fusion_explanation["no_match_threshold"]


# ===========================================================================
# FX5 — Invariants across random scenarios
# ===========================================================================

class TestFX5_Invariants:

    @pytest.mark.parametrize("seed", range(25))
    def test_explanation_consistent_with_match(self, seed):
        rng = np.random.default_rng(seed)
        q, q_face = _base_emb(1000 + seed), _base_emb(2000 + seed)
        qk = _query_kpr(q)
        gallery, kpr_g, face_g = [], [], []
        for tid in range(1, 5):
            gallery += _gallery(q, float(rng.uniform(0.62, 0.98)), track_id=tid)
            if rng.random() < 0.8:
                kpr_g += _kpr_gallery(qk, float(rng.uniform(0.3, 0.98)), track_id=tid)
            if rng.random() < 0.7:
                face_g += _face_gallery(q_face, float(rng.uniform(-0.5, 0.9)), track_id=tid)

        res = _svc.search_camera_top_k(
            query_embedding=q, gallery=gallery, camera_id="C01", candidate_tracks=4,
            kpr_gallery=kpr_g, query_kpr=qk,
            face_gallery=face_g, query_face_embedding=q_face, query_face_confidence=0.95,
        )
        for m in res:
            ex = m.fusion_explanation
            assert round(ex["fused_score"], 6) == m.fused_score
            assert sum(s["contribution"] for s in ex["signals"]) == \
                pytest.approx(ex["fused_score"], abs=1e-12)
            assert ex["face_veto_applied"] == m.face_veto_applied
            if m.match_status == MatchStatus.CONFIDENT_MATCH:
                assert ex["margin_to_threshold"] >= 0
            elif m.match_status == MatchStatus.POSSIBLE_MATCH_REVIEW:
                assert ex["margin_to_threshold"] < 0
            names = [s["signal"] for s in ex["signals"]] + \
                    [e["signal"] for e in ex["excluded_signals"]]
            assert sorted(names) == ["body", "face", "kpr"]
            assert json.loads(json.dumps(ex)) == ex
