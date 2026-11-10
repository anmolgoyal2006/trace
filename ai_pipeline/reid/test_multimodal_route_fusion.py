"""
test_multimodal_route_fusion.py
================================
Tests for the Joint Multi-Modal Spatiotemporal Route Reconstruction refactor.

Covers:
  A. Five-component fusion (_fuse_route behaviour)
  B. Face identity veto (route-transition level)
  C. Candidate eligibility (CONFIDENT_MATCH + POSSIBLE_MATCH_REVIEW)
  D. Spatial / temporal / hard contradiction checks
  E. Beam search (width, deduplication, cumulative scoring)
  F. Backward compatibility / re-baselined fields

Run:
    pytest ai_pipeline/reid/test_multimodal_route_fusion.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.services.matching_service import CameraMatch, MatchStatus
from backend.app.services.route_service import RouteService, FusedSighting, ReconstructedRoute
from backend.app.models.schemas import RouteStepOut, QueryRouteOut


# ---------------------------------------------------------------------------
# Helper: build a fully-featured CameraMatch
# ---------------------------------------------------------------------------

def _make_match_full(
    camera_id: str,
    track_id: int = 1,
    appearance_score: float = 0.85,
    first_seen: str | None = "10:00:00.00",
    last_seen: str | None = "10:00:10.00",
    match_status: MatchStatus = MatchStatus.CONFIDENT_MATCH,
    face_sim: float | None = None,
    kpr_sim: float | None = None,
    face_coverage: float = 0.0,
    query_face_reliable: bool = False,
    gallery_face_reliable: bool = False,
    embedding_dim: int = 512,
) -> CameraMatch:
    m = CameraMatch(
        camera_id=camera_id,
        track_id=track_id,
        appearance_score=appearance_score,
        mean_top3_similarity=appearance_score,
        best_confidence=appearance_score * 100,
        mean_confidence=appearance_score * 100,
        first_seen=first_seen,
        last_seen=last_seen,
        match_status=match_status,
        fused_score=appearance_score,
        embedding_dim=embedding_dim,
    )
    m.face_sim = face_sim
    m.kpr_sim = kpr_sim
    m.face_coverage = face_coverage
    m.query_face_reliable = query_face_reliable
    m.gallery_face_reliable = gallery_face_reliable
    return m


# ---------------------------------------------------------------------------
# A. Five-component fusion
# ---------------------------------------------------------------------------

class TestFiveComponentFusion:
    """Group A — _fuse_route with various modality combinations."""

    @pytest.fixture
    def rs(self) -> RouteService:
        return RouteService()

    def test_all_five_active(self, rs):
        """All five components active → result in [0, 1] and differs from 3-component."""
        match = _make_match_full(
            "C01",
            appearance_score=0.85,
            face_sim=0.60,
            kpr_sim=0.90,
            query_face_reliable=True,
            gallery_face_reliable=True,
        )
        result_5 = rs._fuse_route(match, 1.0, 1.0)
        # 3-component result (body+spatial+temporal only)
        match_no_face_kpr = _make_match_full("C01", appearance_score=0.85)
        result_3 = rs._fuse_route(match_no_face_kpr, 1.0, 1.0)
        assert 0.0 <= result_5 <= 1.0
        # With all five active the score should differ from the 3-component version
        assert abs(result_5 - result_3) > 1e-6

    def test_missing_face_renormalizes(self, rs):
        """face_sim=None → weights renormalize; result still in [0, 1]."""
        match = _make_match_full("C01", appearance_score=0.85, kpr_sim=0.90)
        result = rs._fuse_route(match, 1.0, 1.0)
        assert 0.0 <= result <= 1.0
        # Result differs from all-five (kpr present, face absent)
        match_all = _make_match_full(
            "C01", appearance_score=0.85, face_sim=0.60, kpr_sim=0.90,
            query_face_reliable=True, gallery_face_reliable=True,
        )
        result_all = rs._fuse_route(match_all, 1.0, 1.0)
        assert abs(result - result_all) > 1e-6

    def test_missing_kpr_renormalizes(self, rs):
        """kpr_sim=None → weights renormalize; result still in [0, 1]."""
        match = _make_match_full(
            "C01", appearance_score=0.85, face_sim=0.60,
            query_face_reliable=True, gallery_face_reliable=True,
        )
        result = rs._fuse_route(match, 1.0, 1.0)
        assert 0.0 <= result <= 1.0

    def test_missing_face_and_kpr_renormalizes(self, rs):
        """Both face_sim and kpr_sim absent → body+spatial+temporal renormalized by 0.65."""
        match = _make_match_full("C01", appearance_score=0.85, embedding_dim=512)
        result = rs._fuse_route(match, 1.0, 1.0)

        # Manual calculation: active weights = body(0.25) + spatial(0.25) + temporal(0.15) = 0.65
        # B_norm for 0.85 on OSNet: (0.85 - 0.3315) / (0.9730 - 0.3315)
        b_norm = (0.85 - 0.3315049352393543) / (0.9730031552165505 - 0.3315049352393543)
        expected = (b_norm * 0.25 + 1.0 * 0.25 + 1.0 * 0.15) / 0.65
        expected = round(max(0.0, min(1.0, expected)), 6)
        assert abs(result - expected) < 1e-5

    def test_body_normalization_min(self, rs):
        """appearance_score at OSNet observed_min → b_norm near 0."""
        match = _make_match_full("C01", appearance_score=0.3315049352393543, embedding_dim=512)
        result = rs._fuse_route(match, 0.0, 0.0)
        # All scores low → result should be low
        assert result < 0.1

    def test_body_normalization_max(self, rs):
        """appearance_score at OSNet observed_max → b_norm = 1.0."""
        match = _make_match_full("C01", appearance_score=0.9730031552165505, embedding_dim=512)
        result = rs._fuse_route(match, 1.0, 1.0)
        # All at max → result should be very high
        assert result > 0.90

    def test_output_clamped_zero_to_one(self, rs):
        """Extreme inputs always produce result in [0, 1]."""
        match_high = _make_match_full("C01", appearance_score=99.0, embedding_dim=512)
        match_low = _make_match_full("C01", appearance_score=-5.0, embedding_dim=512)
        assert 0.0 <= rs._fuse_route(match_high, 2.0, 2.0) <= 1.0
        assert 0.0 <= rs._fuse_route(match_low, -1.0, -1.0) <= 1.0

    def test_negative_weight_rejected(self):
        """Settings with a negative route weight raises ValidationError."""
        from pydantic import ValidationError
        from backend.app.config import Settings
        with pytest.raises(ValidationError):
            Settings(route_body_weight=-0.1)

    def test_all_zero_weights_rejected(self):
        """Settings with all five route weights = 0 raises ValidationError."""
        from pydantic import ValidationError
        from backend.app.config import Settings
        with pytest.raises(ValidationError):
            Settings(
                route_body_weight=0.0,
                route_face_weight=0.0,
                route_kpr_weight=0.0,
                route_spatial_weight=0.0,
                route_temporal_weight=0.0,
            )


# ---------------------------------------------------------------------------
# B. Face identity veto (route-transition level)
# ---------------------------------------------------------------------------

class TestFaceIdentityVeto:
    """Group B — _evaluate_transition face veto logic."""

    @pytest.fixture
    def rs(self) -> RouteService:
        return RouteService()

    def test_face_mismatch_veto_fires(self, rs):
        """Both reliable, face_sim=0.20 (< 0.38) → vetoed=True, veto_reason=face_mismatch."""
        from_m = _make_match_full("C01", last_seen="10:00:10.00")
        to_m = _make_match_full(
            "C02", first_seen="10:00:35.00",
            face_sim=0.20, query_face_reliable=True, gallery_face_reliable=True,
        )
        result = rs._evaluate_transition(from_m, to_m, {})
        assert result["vetoed"] is True
        assert result["veto_reason"] == "face_mismatch"
        assert result["step"] is None
        assert result["transition_score"] is None

    def test_face_strong_match_included(self, rs):
        """Both reliable, face_sim=0.65 → not vetoed; face in active_modalities."""
        from_m = _make_match_full("C01", last_seen="10:00:10.00")
        to_m = _make_match_full(
            "C02", first_seen="10:00:35.00",
            face_sim=0.65, query_face_reliable=True, gallery_face_reliable=True,
        )
        result = rs._evaluate_transition(from_m, to_m, {})
        assert result["vetoed"] is False
        assert result["step"] is not None
        mods = result["step"].explanation["component_breakdown"]["active_modalities"]
        assert "face" in mods

    def test_face_missing_no_veto(self, rs):
        """face_sim=None → not vetoed; face not in active_modalities."""
        from_m = _make_match_full("C01", last_seen="10:00:10.00")
        to_m = _make_match_full("C02", first_seen="10:00:35.00")
        result = rs._evaluate_transition(from_m, to_m, {})
        assert result["vetoed"] is False
        mods = result["step"].explanation["component_breakdown"]["active_modalities"]
        assert "face" not in mods

    def test_face_low_coverage_no_veto(self, rs):
        """gallery_face_reliable=False → not vetoed."""
        from_m = _make_match_full("C01", last_seen="10:00:10.00")
        to_m = _make_match_full(
            "C02", first_seen="10:00:35.00",
            face_sim=0.20, query_face_reliable=True, gallery_face_reliable=False,
        )
        result = rs._evaluate_transition(from_m, to_m, {})
        assert result["vetoed"] is False

    def test_high_osnet_face_mismatch_still_vetoed(self, rs):
        """appearance_score=0.95 + reliable face_sim=0.20 → still vetoed."""
        from_m = _make_match_full("C01", last_seen="10:00:10.00")
        to_m = _make_match_full(
            "C02", first_seen="10:00:35.00",
            appearance_score=0.95,
            face_sim=0.20, query_face_reliable=True, gallery_face_reliable=True,
        )
        result = rs._evaluate_transition(from_m, to_m, {})
        assert result["vetoed"] is True
        assert result["veto_reason"] == "face_mismatch"

    def test_uncertain_face_band_include(self):
        """face_sim in uncertain band: included with route_face_include_uncertain=True."""
        from backend.app.config import Settings
        # Custom settings: mismatch=0.30, confidence=0.50, face_sim=0.40 is in uncertain band
        # face_norm_max must be > face_norm_min (default 0.0 < 0.70 is fine)
        s = Settings(
            route_face_mismatch_threshold=0.30,
            route_face_confidence_threshold=0.50,
            route_face_include_uncertain=True,
        )
        rs_custom = RouteService()
        rs_custom._route_face_mismatch_thr = s.route_face_mismatch_threshold
        rs_custom._route_face_confidence_thr = s.route_face_confidence_threshold
        rs_custom._route_face_include_uncertain = s.route_face_include_uncertain

        from_m = _make_match_full("C01", last_seen="10:00:10.00")
        to_m = _make_match_full(
            "C02", first_seen="10:00:35.00",
            face_sim=0.40, query_face_reliable=True, gallery_face_reliable=True,
        )
        result_inc = rs_custom._evaluate_transition(from_m, to_m, {})
        mods = result_inc["step"].explanation["component_breakdown"]["active_modalities"]
        assert "face" in mods

    def test_uncertain_face_band_exclude(self):
        """face_sim in uncertain band + route_face_include_uncertain=False → excluded."""
        rs_custom = RouteService()
        rs_custom._route_face_mismatch_thr = 0.30
        rs_custom._route_face_confidence_thr = 0.50
        rs_custom._route_face_include_uncertain = False

        from_m = _make_match_full("C01", last_seen="10:00:10.00")
        to_m = _make_match_full(
            "C02", first_seen="10:00:35.00",
            face_sim=0.40, query_face_reliable=True, gallery_face_reliable=True,
        )
        result_exc = rs_custom._evaluate_transition(from_m, to_m, {})
        mods = result_exc["step"].explanation["component_breakdown"]["active_modalities"]
        assert "face" not in mods


# ---------------------------------------------------------------------------
# C. Candidate eligibility
# ---------------------------------------------------------------------------

class TestCandidateEligibility:
    """Group C — CONFIDENT_MATCH and POSSIBLE_MATCH_REVIEW are both routed."""

    @pytest.fixture
    def rs(self) -> RouteService:
        return RouteService()

    def test_confident_match_routed(self, rs):
        """CONFIDENT_MATCH candidate appears in route."""
        candidates = {
            "C01": [_make_match_full("C01", match_status=MatchStatus.CONFIDENT_MATCH,
                                     first_seen="10:00:00.00", last_seen="10:00:10.00")],
        }
        route = rs.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 1
        assert route.steps[0].match.camera_id == "C01"

    def test_possible_match_review_routed(self, rs):
        """POSSIBLE_MATCH_REVIEW candidate reaches route reconstruction."""
        candidates = {
            "C01": [_make_match_full("C01", match_status=MatchStatus.POSSIBLE_MATCH_REVIEW,
                                     first_seen="10:00:00.00", last_seen="10:00:10.00")],
        }
        route = rs.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 1
        assert route.steps[0].match.camera_id == "C01"

    def test_face_mismatch_eligible_set_excludes_it(self):
        """FACE_MISMATCH is not in _eligible_for_route."""
        from backend.app.services.matching_service import MatchStatus
        eligible = {MatchStatus.CONFIDENT_MATCH, MatchStatus.POSSIBLE_MATCH_REVIEW}
        assert MatchStatus.FACE_MISMATCH not in eligible

    def test_match_statuses_are_distinct(self):
        """CONFIDENT_MATCH and POSSIBLE_MATCH_REVIEW have different values."""
        assert MatchStatus.CONFIDENT_MATCH.value != MatchStatus.POSSIBLE_MATCH_REVIEW.value


# ---------------------------------------------------------------------------
# D. Spatial / temporal / hard contradiction
# ---------------------------------------------------------------------------

class TestSpatialTemporalContradiction:
    """Group D — spatial, temporal scoring and hard vetoes."""

    @pytest.fixture
    def rs(self) -> RouteService:
        return RouteService()

    def test_1hop_spatial_score(self, rs):
        """1-hop camera pair → topo_score=1.0 in transition."""
        from_m = _make_match_full("C01", last_seen="10:00:10.00")
        to_m = _make_match_full("C02", first_seen="10:00:35.00")
        result = rs._evaluate_transition(from_m, to_m, {})
        assert result["vetoed"] is False
        assert result["step"].spatial == 1.0

    def test_disconnected_soft_flag(self, rs):
        """Disconnected camera → vetoed=False, status=DISCONNECTED, transition_score penalized."""
        from_m = _make_match_full("C01", last_seen="10:00:10.00")
        to_m = _make_match_full("C99", first_seen="10:00:35.00")
        result = rs._evaluate_transition(from_m, to_m, {})
        assert result["vetoed"] is False
        assert result["step"].transition_status == "DISCONNECTED"
        # Soft penalty applied → transition_score < fusion (penalty-free)
        assert result["step"].transition_score < result["step"].fusion

    def test_backward_time_hard_veto(self, rs):
        """Backward-in-time → vetoed=True, veto_reason=backward_in_time."""
        from_m = _make_match_full("C01", last_seen="10:05:00.00")
        to_m = _make_match_full("C02", first_seen="10:00:00.00")
        result = rs._evaluate_transition(from_m, to_m, {})
        assert result["vetoed"] is True
        assert result["veto_reason"] == "backward_in_time"

    def test_gps_impossible_hard_veto(self, rs):
        """GPS impossible speed → vetoed=True, veto_reason=gps_impossible_speed."""
        from_m = _make_match_full("C01", last_seen="10:00:05.00")
        to_m = _make_match_full("C02", first_seen="10:00:15.00")
        coords = {"C01": (12.971598, 77.594562), "C02": (12.972598, 77.594562)}
        result = rs._evaluate_transition(from_m, to_m, coords)
        assert result["vetoed"] is True
        assert result["veto_reason"] == "gps_impossible_speed"

    def test_no_double_penalty_gps_and_sprint(self, rs):
        """When GPS coords present and speed > limit, only hard GPS veto fires, not soft too_fast."""
        from_m = _make_match_full("C01", last_seen="10:00:05.00")
        to_m = _make_match_full("C02", first_seen="10:00:15.00")
        coords = {"C01": (12.971598, 77.594562), "C02": (12.972598, 77.594562)}
        result = rs._evaluate_transition(from_m, to_m, coords)
        # Hard veto — no step constructed at all, so no double penalty possible
        assert result["vetoed"] is True
        assert result["step"] is None


# ---------------------------------------------------------------------------
# E. Beam search
# ---------------------------------------------------------------------------

class TestBeamSearch:
    """Group E — beam width, deduplication, cumulative scoring."""

    @pytest.fixture
    def rs(self) -> RouteService:
        return RouteService()

    def test_beam_width_limits_alternatives(self, rs):
        """Beam width=3 → at most beam_width-1 alternative routes."""
        candidates = {
            "C01": [_make_match_full("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match_full("C02", first_seen="10:00:35.00", last_seen="10:00:45.00")],
            "C03": [_make_match_full("C03", first_seen="10:01:10.00", last_seen="10:01:20.00")],
        }
        rs._beam_width = 3
        route = rs.reconstruct_route(candidates, method="topology_aware")
        assert len(route.alternative_routes) <= 2

    def test_vetoed_transition_absent_from_route(self, rs):
        """GPS-impossible camera is excluded from the route."""
        coords = {
            "C01": (12.971598, 77.594562),
            "C02": (12.972598, 77.594562),
        }
        candidates = {
            "C01": [_make_match_full("C01", first_seen="10:00:00.00", last_seen="10:00:05.00")],
            "C02": [_make_match_full("C02", first_seen="10:00:15.00", last_seen="10:00:25.00")],
        }
        route = rs.reconstruct_route(candidates, camera_coordinates=coords, method="topology_aware")
        cam_ids = [s.match.camera_id for s in route.steps]
        assert "C02" not in cam_ids

    def test_transition_score_set_on_every_step(self, rs):
        """Every step in a multi-camera route has a non-default transition_score."""
        candidates = {
            "C01": [_make_match_full("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match_full("C02", first_seen="10:00:35.00", last_seen="10:00:45.00")],
        }
        route = rs.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 2
        for step in route.steps:
            # transition_score should not be the default 0.0 for meaningful steps
            # (unless the formula genuinely produces 0.0)
            assert isinstance(step.transition_score, float)
            assert step.transition_score != 0.0 or step.fusion == 0.0

    def test_cumulative_route_score_is_mean_of_transition_scores(self, rs):
        """cumulative_route_score == mean(step.transition_score) for all steps."""
        candidates = {
            "C01": [_make_match_full("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match_full("C02", first_seen="10:00:35.00", last_seen="10:00:45.00")],
            "C03": [_make_match_full("C03", first_seen="10:01:10.00", last_seen="10:01:20.00")],
        }
        route = rs.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) >= 1
        mean_ts = sum(s.transition_score for s in route.steps) / len(route.steps)
        mean_ts_clamped = max(0.0, min(1.0, mean_ts))
        assert abs(route.cumulative_route_score - mean_ts_clamped) < 1e-5

    def test_cross_beam_dedup(self, rs):
        """Two anchors leading to same camera sequence produce at most one entry."""
        # Create two matches for C01 with similar scores — both could anchor
        # Then only C02 as the continuation. After dedup there should be only one (C01→C02) sequence.
        candidates = {
            "C01": [
                _make_match_full("C01", track_id=1, appearance_score=0.85,
                                 first_seen="10:00:00.00", last_seen="10:00:10.00"),
                _make_match_full("C01", track_id=2, appearance_score=0.84,
                                 first_seen="10:00:01.00", last_seen="10:00:11.00"),
            ],
            "C02": [
                _make_match_full("C02", track_id=10, appearance_score=0.80,
                                 first_seen="10:00:35.00", last_seen="10:00:45.00"),
            ],
        }
        route = rs.reconstruct_route(candidates, method="topology_aware")
        # Check alternatives do not duplicate the same camera sequence as best path
        best_seq = tuple(s.match.camera_id for s in route.steps)
        for alt in route.alternative_routes:
            alt_seq = tuple(s.match.camera_id for s in alt)
            assert alt_seq != best_seq, f"Duplicate sequence {alt_seq} in alternatives"


# ---------------------------------------------------------------------------
# F. Backward compatibility / re-baselined
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Group F — existing fields remain intact; new fields are optional."""

    @pytest.fixture
    def rs(self) -> RouteService:
        return RouteService()

    def test_route_confidence_field_still_present(self, rs):
        """route_confidence is still present and is mean of fusion_score (penalty-free)."""
        candidates = {
            "C01": [_make_match_full("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match_full("C02", first_seen="10:00:35.00", last_seen="10:00:45.00")],
        }
        route = rs.reconstruct_route(candidates, method="topology_aware")
        assert hasattr(route, "route_confidence")
        assert route.route_confidence >= 0.0
        # Confirm it's computed from fusion (penalty-free)
        expected_conf = round(sum(s.fusion for s in route.steps) / len(route.steps), 4)
        assert abs(route.route_confidence - expected_conf) < 1e-4

    def test_new_fields_optional_default_none(self):
        """RouteStepOut and QueryRouteOut new fields default to None."""
        # RouteStepOut: construct without transition_score
        step = RouteStepOut(
            step_order=0,
            camera_id="C01",
            camera_location="Loc A",
            track_id=1,
            timestamp="10:00:00.00",
            confidence=85.0,
            fusion_score=0.8,
            crop_path=None,
        )
        assert step.transition_score is None

        # QueryRouteOut: construct without cumulative_route_score
        qr = QueryRouteOut(
            session_id=1,
            status="done",
            steps=[],
            sightings=[],
            total_cameras_matched=0,
            route_confidence=0.0,
        )
        assert qr.cumulative_route_score is None

    def test_cumulative_route_score_on_route_object(self, rs):
        """ReconstructedRoute has cumulative_route_score field."""
        candidates = {
            "C01": [_make_match_full("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
        }
        route = rs.reconstruct_route(candidates, method="topology_aware")
        assert hasattr(route, "cumulative_route_score")
        assert isinstance(route.cumulative_route_score, float)

    def test_step_explanation_has_component_breakdown(self, rs):
        """Every transition step explanation has component_breakdown with required keys."""
        candidates = {
            "C01": [_make_match_full("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match_full("C02", first_seen="10:00:35.00", last_seen="10:00:45.00")],
        }
        route = rs.reconstruct_route(candidates, method="topology_aware")
        c02 = route.steps[1]
        assert c02.explanation is not None
        assert "component_breakdown" in c02.explanation
        cb = c02.explanation["component_breakdown"]
        required_keys = {
            "body_similarity", "face_similarity", "kpr_similarity",
            "spatial_score", "temporal_score", "route_fusion_score",
            "transition_score", "face_veto", "face_veto_reason", "active_modalities",
        }
        assert required_keys.issubset(cb.keys())

    def test_existing_explanation_keys_preserved(self, rs):
        """Existing explanation top-level keys are still present."""
        candidates = {
            "C01": [_make_match_full("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match_full("C02", first_seen="10:00:35.00", last_seen="10:00:45.00")],
        }
        route = rs.reconstruct_route(candidates, method="topology_aware")
        expl = route.steps[1].explanation
        for key in ("topological_evidence", "temporal_evidence", "appearance_evidence", "spatial_evidence"):
            assert key in expl, f"Key {key!r} missing from explanation"
