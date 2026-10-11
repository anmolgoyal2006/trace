"""
test_fused_score_regression.py — XAI Milestone 1: Fusion Arithmetic Regression
===============================================================================
Locks down the numeric behaviour of the _fused_score closure inside
MatchingService.search_camera_top_k (matching_service.py lines 591-630).

test_matching_service.py tests outcomes (CONFIDENT_MATCH / FACE_MISMATCH).
These tests verify the *arithmetic* — the exact weighted-average formula — so
any silent formula change is caught before it affects rankings or thresholds.

Formula (matching_service.py lines 607-630):
    if not use_face and not use_kpr:
        return body_sim                            # fast path
    num, den = body_w * body_sim, body_w
    if use_kpr and kpr_sim is not None:
        num += kpr_w * kpr_sim;  den += kpr_w
    if use_face and face_sim is not None and face_sim > 0:
        num += face_w * face_sim; den += face_w
    return float(num / den)

Weights (backend/app/config.py defaults):
    fusion_body_weight = 0.40   fusion_face_weight = 0.30   fusion_kpr_weight = 0.60

Coverage:
    FS8  weight constants unchanged (run first — all arithmetic tests depend on this)
    FS1  body-only fast path: fused_score == body_sim, no division
    FS2  body + KPR: (0.40*b + 0.60*k) / 1.00, denominator excludes absent signals
    FS3  all three signals: (0.40*b + 0.60*k + 0.30*f) / 1.30
    FS4  KPR gallery present but kpr_sim=None: kpr_w dropped from denominator
    FS5  positive face_sim: enters both numerator and denominator
    FS6  face_sim == 0: excluded (> 0 guard), denominator not inflated
    FS7  face_sim < 0: excluded from arithmetic; veto fires separately

Run:
    pytest ai_pipeline/reid/test_fused_score_regression.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ai_pipeline.reid.similarity import cosine_similarity
from backend.app.config import settings
from backend.app.services.matching_service import MatchingService

# ---------------------------------------------------------------------------
# Production weights — read once from settings
# ---------------------------------------------------------------------------
_BW = settings.fusion_body_weight   # 0.40
_FW = settings.fusion_face_weight   # 0.30
_KW = settings.fusion_kpr_weight    # 0.60

_svc = MatchingService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_emb(seed: int) -> list[float]:
    """Reproducible unit-norm 512-dim embedding."""
    v = np.random.default_rng(seed).standard_normal(512)
    return (v / np.linalg.norm(v)).tolist()


def _emb_with_sim(query: list[float], target_sim: float, seed: int = 99) -> list[float]:
    """
    Return a unit-norm vector g with cosine_similarity(query, g) == target_sim exactly.

    Geometric construction: g = t*q̂ + √(1−t²)*perp, where perp is a unit
    vector orthogonal to query (Gram-Schmidt). Error < 3e-16 (machine epsilon).
    """
    q = np.array(query)
    q_hat = q / np.linalg.norm(q)
    r = np.random.default_rng(seed).standard_normal(len(q))
    r -= (r @ q_hat) * q_hat
    r /= np.linalg.norm(r)
    g = target_sim * q_hat + math.sqrt(max(0.0, 1.0 - target_sim ** 2)) * r
    return (g / np.linalg.norm(g)).tolist()


def _gallery(emb: list[float], track_id: int = 1) -> list[dict]:
    return [{"embedding": emb, "track_id": track_id,
             "crop_path": f"crops/t{track_id}_f0.jpg",
             "timestamp": "10:00:01.000", "frame": 1}]


def _face_gallery(face_emb: list[float], track_id: int = 1) -> list[dict]:
    """Three crops with identical face_emb → centroid == face_emb exactly."""
    return [{"track_id": track_id, "face_detected": True, "face_embedding": face_emb,
             "face_confidence": 0.92, "crop_path": f"crops/t{track_id}_f{i}.jpg",
             "camera_id": "C01", "frame": i, "timestamp": f"10:00:0{i}", "bbox": None}
            for i in range(3)]


def _kpr_gallery(q_kpr: dict, track_id: int = 1, part_sim: float = 0.92) -> list[dict]:
    parts = [_emb_with_sim(p, part_sim, seed=i + track_id * 100)
             for i, p in enumerate(q_kpr["part_embeddings"])]
    return [{"track_id": track_id, "crop_path": f"crops/kpr{track_id}_f0.jpg",
             "timestamp": "10:00:01.000", "frame": 1,
             "holistic_embedding": q_kpr["holistic_embedding"],
             "part_embeddings": parts, "part_visibility": [0.95] * len(parts),
             "camera_id": "C01", "bbox": None, "detection_confidence": 0.91}]


def _query_kpr(base: list[float], n_parts: int = 5) -> dict:
    parts = [_emb_with_sim(base, 0.99, seed=i + 200) for i in range(n_parts)]
    return {"holistic_embedding": base,
            "part_embeddings": parts, "part_visibility": [0.95] * n_parts}


def _expected(body: float, kpr: float | None = None, face: float | None = None) -> float:
    """Reference implementation of _fused_score, transcribed verbatim from source."""
    if kpr is None and face is None:
        return body
    num, den = _BW * body, _BW
    if kpr is not None:
        num += _KW * kpr;  den += _KW
    if face is not None and face > 0:
        num += _FW * face; den += _FW
    return float(num / den)


def _search(q, gal, *, face_gal=None, q_face=None, q_face_conf=None,
            kpr_gal=None, q_kpr=None):
    return _svc.search_camera_top_k(
        query_embedding=q, gallery=gal, camera_id="C01",
        face_gallery=face_gal, query_face_embedding=q_face,
        query_face_confidence=q_face_conf,
        kpr_gallery=kpr_gal, query_kpr=q_kpr,
    )


# ===========================================================================
# FS8 — Weight constants unchanged (run first; arithmetic tests depend on this)
# ===========================================================================

class TestFS8_WeightConstants:
    def test_weights_match_config_and_service(self):
        assert _BW == pytest.approx(0.40), f"fusion_body_weight changed to {_BW}"
        assert _FW == pytest.approx(0.30), f"fusion_face_weight changed to {_FW}"
        assert _KW == pytest.approx(0.60), f"fusion_kpr_weight changed to {_KW}"
        assert _svc._body_weight == pytest.approx(_BW)
        assert _svc._face_weight == pytest.approx(_FW)
        assert _svc._kpr_weight  == pytest.approx(_KW)


# ===========================================================================
# FS1 — Body-only fast path: fused_score == body_sim, no division
# ===========================================================================

class TestFS1_BodyOnlyFastPath:
    """_fused_score returns body_sim directly when no optional galleries are given."""

    def test_fused_equals_body_sim(self):
        q, g = _base_emb(1), _emb_with_sim(_base_emb(1), 0.90, seed=10)
        res  = _search(q, _gallery(g))
        assert res, "body_sim=0.90 must clear soft_floor=0.60"
        assert res[0].fused_score == pytest.approx(cosine_similarity(q, g), abs=1e-6)

    def test_explicit_none_args_use_fast_path(self):
        q, g = _base_emb(2), _emb_with_sim(_base_emb(2), 0.85, seed=20)
        res  = _search(q, _gallery(g), face_gal=None, q_face=None, kpr_gal=None, q_kpr=None)
        assert res
        assert res[0].fused_score == pytest.approx(cosine_similarity(q, g), abs=1e-6)

    def test_fused_equals_appearance_score(self):
        q, g = _base_emb(3), _emb_with_sim(_base_emb(3), 0.80, seed=30)
        res  = _search(q, _gallery(g))
        assert res
        assert res[0].fused_score == pytest.approx(res[0].appearance_score, abs=1e-6)


# ===========================================================================
# FS2 — Body + KPR: (0.40*body + 0.60*kpr) / 1.00
# ===========================================================================

class TestFS2_BodyKPR:
    """Denominator == body_w + kpr_w == 1.00 when only these two signals are active."""

    def test_body_kpr_arithmetic(self):
        q    = _base_emb(4)
        g    = _emb_with_sim(q, 0.80, seed=40)
        body = cosine_similarity(q, g)

        q_kpr = _query_kpr(q)
        res   = _search(q, _gallery(g), kpr_gal=_kpr_gallery(q_kpr, part_sim=0.90), q_kpr=q_kpr)
        assert res
        m = res[0]
        assert m.kpr_sim is not None, "kpr_sim must be populated"

        expected = _expected(body, kpr=m.kpr_sim)
        assert m.fused_score == pytest.approx(expected, abs=1e-5), (
            f"(0.40*{body:.4f}+0.60*{m.kpr_sim:.4f})/1.00={expected:.6f}, got {m.fused_score:.6f}"
        )

    def test_kpr_raises_fused_above_body_sim(self):
        """kpr_sim > body_sim → fused must be > body_sim (blending is active)."""
        q    = _base_emb(5)
        g    = _emb_with_sim(q, 0.72, seed=50)
        body = cosine_similarity(q, g)

        q_kpr = _query_kpr(q)
        res   = _search(q, _gallery(g), kpr_gal=_kpr_gallery(q_kpr, part_sim=0.95), q_kpr=q_kpr)
        assert res
        m = res[0]
        assert m.kpr_sim is not None
        if m.kpr_sim > body:
            assert m.fused_score > body, (
                f"kpr_sim={m.kpr_sim:.4f}>body={body:.4f} but fused={m.fused_score:.4f} did not increase"
            )


# ===========================================================================
# FS3 — All three signals: (0.40*b + 0.60*k + 0.30*f) / 1.30
# ===========================================================================

class TestFS3_AllThreeSignals:
    """Denominator == body_w + kpr_w + face_w == 1.30 when all signals are active."""

    def test_three_signal_arithmetic(self):
        q     = _base_emb(7)
        g     = _emb_with_sim(q, 0.82, seed=70)
        body  = cosine_similarity(q, g)
        qf    = _base_emb(8)
        gf    = _emb_with_sim(qf, 0.85, seed=80)
        q_kpr = _query_kpr(q)

        res = _search(q, _gallery(g),
                      face_gal=_face_gallery(gf), q_face=qf, q_face_conf=0.95,
                      kpr_gal=_kpr_gallery(q_kpr, part_sim=0.91), q_kpr=q_kpr)
        assert res

        m = res[0]
        assert m.kpr_sim  is not None, "kpr_sim must be set"
        assert m.face_sim is not None and m.face_sim > 0, "face_sim must be positive"

        expected = _expected(body, kpr=m.kpr_sim, face=m.face_sim)
        assert m.fused_score == pytest.approx(expected, abs=1e-5), (
            f"(0.40*{body:.4f}+0.60*{m.kpr_sim:.4f}+0.30*{m.face_sim:.4f})"
            f"/1.30={expected:.6f}, got {m.fused_score:.6f}"
        )


# ===========================================================================
# FS4 — KPR gallery present but kpr_sim=None: kpr_w dropped from denominator
# ===========================================================================

class TestFS4_KPRMissing:
    """
    KPR gallery exists but has no record for the matched track.
    The ``if k_sim is not None:`` guard must prevent kpr_w from entering
    the denominator; fused collapses to body_sim.
    """

    def test_kpr_none_collapses_to_body_sim(self):
        q    = _base_emb(12)
        g    = _emb_with_sim(q, 0.85, seed=120)
        body = cosine_similarity(q, g)

        q_kpr   = _query_kpr(q)
        # KPR gallery belongs to track_id=99, not the gallery's track_id=1
        kpr_gal = _kpr_gallery(q_kpr, track_id=99, part_sim=0.95)

        res = _search(q, _gallery(g, track_id=1), kpr_gal=kpr_gal, q_kpr=q_kpr)
        assert res

        m = res[0]
        assert m.kpr_sim is None, "No KPR record for track=1 → kpr_sim must be None"

        # Correct: denominator stays body_w (0.40) → fused == body_sim
        # Wrong:   denominator inflated to body_w+kpr_w (1.00) → fused == 0.40*body
        wrong = (_BW * body) / (_BW + _KW)
        assert m.fused_score == pytest.approx(body, abs=1e-5), (
            f"kpr_sim=None: fused must equal body={body:.6f}, "
            f"wrong (inflated denom) would be {wrong:.6f}, got {m.fused_score:.6f}"
        )
        assert m.fused_score != pytest.approx(wrong, abs=1e-4)


# ===========================================================================
# FS5 — Positive face_sim: enters both numerator and denominator
# ===========================================================================

class TestFS5_PositiveFaceIncluded:
    """face_sim > 0 → fused = (0.40*body + 0.30*face) / 0.70"""

    def test_positive_face_arithmetic(self):
        q    = _base_emb(14)
        g    = _emb_with_sim(q, 0.80, seed=140)
        body = cosine_similarity(q, g)
        qf   = _base_emb(15)
        gf   = _emb_with_sim(qf, 0.82, seed=150)

        res = _search(q, _gallery(g), face_gal=_face_gallery(gf), q_face=qf, q_face_conf=0.95)
        assert res
        m = res[0]
        assert m.face_sim is not None and m.face_sim > 0, "face_sim must be positive"

        expected = _expected(body, face=m.face_sim)   # (0.40*b + 0.30*f) / 0.70
        assert m.fused_score == pytest.approx(expected, abs=1e-5), (
            f"(0.40*{body:.4f}+0.30*{m.face_sim:.4f})/0.70={expected:.6f}, got {m.fused_score:.6f}"
        )


# ===========================================================================
# FS6 — face_sim == 0: excluded from both numerator and denominator
# ===========================================================================

class TestFS6_ZeroFaceExcluded:
    """
    face_sim == 0 fails ``f_sim > 0``. Face must not enter the formula at all.
    Adding face_w to the denominator when face_sim=0 would give
    fused = (0.40*body) / 0.70 < body_sim — detectable wrong value.
    """

    def test_zero_face_excluded(self):
        q    = _base_emb(18)
        g    = _emb_with_sim(q, 0.85, seed=180)
        body = cosine_similarity(q, g)

        # Standard-basis vectors → cosine_similarity == 0.0 exactly
        qf = [0.0] * 512; qf[0] = 1.0
        gf = [0.0] * 512; gf[1] = 1.0
        assert cosine_similarity(qf, gf) == pytest.approx(0.0, abs=1e-12)

        res = _search(q, _gallery(g), face_gal=_face_gallery(gf), q_face=qf, q_face_conf=0.95)
        assert res

        wrong = (_BW * body) / (_BW + _FW)   # inflated-denominator value
        assert res[0].fused_score == pytest.approx(body, abs=1e-5), (
            f"face_sim=0 must be excluded: correct={body:.6f}, "
            f"wrong (inflated denom)={wrong:.6f}, got {res[0].fused_score:.6f}"
        )
        assert res[0].fused_score != pytest.approx(wrong, abs=1e-4)


# ===========================================================================
# FS7 — face_sim < 0: excluded from arithmetic; veto fires separately
# ===========================================================================

class TestFS7_NegativeFaceExcluded:
    """
    face_sim < 0 fails ``f_sim > 0``. The design intentionally excludes negative
    face from fused_score so a high-body-sim track cannot be dragged below
    soft_floor before the veto check runs (matching_service.py lines 620-628).
    The veto fires separately via _check_face_veto.
    """

    def test_negative_face_excluded_from_fused_score(self):
        q    = _base_emb(20)
        g    = _emb_with_sim(q, 0.88, seed=200)
        body = cosine_similarity(q, g)

        qf = _base_emb(21)
        gf = _emb_with_sim(qf, -0.90, seed=210)
        assert cosine_similarity(qf, gf) < 0

        res = _search(q, _gallery(g), face_gal=_face_gallery(gf), q_face=qf, q_face_conf=0.95)
        assert res

        m     = res[0]
        # Correct: face excluded → fused == body_sim
        # Wrong:   (0.40*body + 0.30*(−0.9)) / 0.70 < body — detectable
        wrong = (_BW * body + _FW * (m.face_sim or -0.9)) / (_BW + _FW)
        assert m.fused_score == pytest.approx(body, abs=1e-5), (
            f"Negative face must be excluded: correct={body:.6f}, "
            f"wrong={wrong:.6f}, got {m.fused_score:.6f}"
        )
        assert m.fused_score != pytest.approx(wrong, abs=1e-4)

    def test_negative_face_still_triggers_veto(self):
        """Veto mechanism is independent of fused_score arithmetic."""
        q  = _base_emb(24)
        g  = _emb_with_sim(q, 0.90, seed=240)
        qf = _base_emb(25)
        gf = _emb_with_sim(qf, -0.90, seed=250)

        res = _search(q, _gallery(g), face_gal=_face_gallery(gf), q_face=qf, q_face_conf=0.95)
        assert res

        m = res[0]
        if m.face_sim is not None and m.face_sim < settings.face_match_threshold:
            assert m.face_veto_applied is True, (
                "face_sim below threshold must trigger veto even though "
                "it is excluded from fused_score arithmetic"
            )
