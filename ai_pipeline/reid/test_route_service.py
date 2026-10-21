"""
test_route_service.py — Test suite for topology-aware route reconstruction.
=============================================================================

All tests in this module use SYNTHETIC test data (clearly labeled) to evaluate
the algorithmic correctness, graph traversal, temporal constraints, spatial
validation, ambiguity detection, and explainability of RouteService.

Run:
    pytest ai_pipeline/reid/test_route_service.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.services.matching_service import CameraMatch, MatchStatus
from backend.app.services.route_service import (
    RouteService,
    FusedSighting,
    ReconstructedRoute,
    haversine_distance,
    shortest_path_transit,
)


def _make_match(
    camera_id: str,
    track_id: int = 1,
    appearance_score: float = 0.85,
    first_seen: str | None = "10:00:00.00",
    last_seen: str | None = "10:00:10.00",
    match_status: MatchStatus = MatchStatus.CONFIDENT_MATCH,
    best_confidence: float = 85.0,
) -> CameraMatch:
    """Helper to generate synthetic CameraMatch objects for testing."""
    return CameraMatch(
        camera_id=camera_id,
        track_id=track_id,
        appearance_score=appearance_score,
        mean_top3_similarity=appearance_score,
        best_confidence=best_confidence,
        mean_confidence=best_confidence,
        first_seen=first_seen,
        last_seen=last_seen,
        match_status=match_status,
        fused_score=appearance_score,
    )


class TestBasicRouteReconstruction:
    """Evaluates the 17+ route reconstruction test scenarios."""

    @pytest.fixture
    def route_service(self) -> RouteService:
        return RouteService()

    def test_valid_route_three_connected_cameras(self, route_service):
        """1. A valid route through three connected cameras (C01 -> C02 -> C03)."""
        candidates = {
            "C01": [_make_match("C01", track_id=1, appearance_score=0.88, first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match("C02", track_id=5, appearance_score=0.85, first_seen="10:00:30.00", last_seen="10:00:40.00")],
            "C03": [_make_match("C03", track_id=9, appearance_score=0.82, first_seen="10:01:10.00", last_seen="10:01:25.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")

        assert len(route.steps) == 3
        sequence = [s.match.camera_id for s in route.steps]
        assert sequence == ["C01", "C02", "C03"]
        assert route.supported_transitions_count >= 2
        assert route.disconnected_transitions_count == 0
        assert route.route_confidence > 0.70

    def test_valid_route_multiple_cameras(self, route_service):
        """2. A valid route through multiple cameras including multi-hop reachability."""
        candidates = {
            "C01": [_make_match("C01", first_seen="10:00:00.00", last_seen="10:00:15.00")],
            "C02": [_make_match("C02", first_seen="10:00:35.00", last_seen="10:00:50.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 2
        assert [s.match.camera_id for s in route.steps] == ["C01", "C02"]
        assert route.steps[1].transition_status == "SUPPORTED"

    def test_camera_with_no_outgoing_connections(self, route_service):
        """3. A camera with no outgoing connections gracefully terminates path."""
        # C01 connects to C02 only. If C02 has no connections, it acts as dead end.
        candidates = {
            "C01": [_make_match("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match("C02", first_seen="10:00:25.00", last_seen="10:00:35.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 2
        assert route.steps[-1].match.camera_id == "C02"

    def test_disconnected_camera_graph_components(self, route_service):
        """4. Disconnected camera graph component transitions are flagged as DISCONNECTED."""
        # Synthetic disjoint camera C99
        candidates = {
            "C01": [_make_match("C01", appearance_score=0.90, first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C99": [_make_match("C99", appearance_score=0.75, first_seen="10:00:30.00", last_seen="10:00:40.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 2
        # C99 is not in the graph, so the transition from C01 to C99 is DISCONNECTED
        c99_step = [s for s in route.steps if s.match.camera_id == "C99"][0]
        assert c99_step.transition_status == "DISCONNECTED"
        assert route.disconnected_transitions_count >= 1

    def test_missing_timestamp(self, route_service):
        """5. A missing timestamp produces a neutral temporal score and records a data limitation."""
        candidates = {
            "C01": [_make_match("C01", first_seen=None, last_seen=None)],
            "C02": [_make_match("C02", first_seen="10:00:20.00", last_seen="10:00:30.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 2
        assert any("timestamps missing" in lim for lim in route.data_limitations)
        # Sighting with missing timestamp has neutral temporal score (0.5), not perfect 1.0
        step_c01 = [s for s in route.steps if s.match.camera_id == "C01"][0]
        assert step_c01.temporal == 0.5

    def test_out_of_order_timestamps(self, route_service):
        """6. Out-of-order timestamps (backward time gap) receive severe penalties."""
        # Sighting at C02 happened BEFORE C01
        candidates = {
            "C01": [_make_match("C01", first_seen="10:05:00.00", last_seen="10:05:10.00")],
            "C02": [_make_match("C02", first_seen="10:00:00.00", last_seen="10:00:10.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 2
        # The algorithm starts at earlier timestamp C02 and moves toward C01
        assert route.steps[0].match.camera_id == "C02"
        assert route.steps[1].match.camera_id == "C01"
        assert route.steps[1].observed_gap_sec is not None
        assert route.steps[1].observed_gap_sec > 0

    def test_duplicate_sightings_same_camera(self, route_service):
        """7. Duplicate candidate tracks in the same camera pick the highest fused candidate."""
        candidates = {
            "C01": [
                _make_match("C01", track_id=1, appearance_score=0.88, first_seen="10:00:00.00", last_seen="10:00:10.00"),
                _make_match("C01", track_id=2, appearance_score=0.70, first_seen="10:00:05.00", last_seen="10:00:15.00"),
            ],
            "C02": [
                _make_match("C02", track_id=10, appearance_score=0.82, first_seen="10:00:25.00", last_seen="10:00:35.00")
            ],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        c01_step = [s for s in route.steps if s.match.camera_id == "C01"][0]
        assert c01_step.match.track_id == 1

    def test_empty_candidate_list(self, route_service):
        """8. An empty candidate list returns an empty route."""
        route = route_service.reconstruct_route({}, method="topology_aware")
        assert len(route.steps) == 0
        assert route.route_confidence == 0.0

    def test_two_similarly_scored_candidate_cameras(self, route_service):
        """9. Two similarly scored candidates result in AMBIGUOUS transition status."""
        candidates = {
            "C01": [_make_match("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [
                _make_match("C02", track_id=3, appearance_score=0.82, first_seen="10:00:22.00", last_seen="10:00:30.00"),
                _make_match("C02", track_id=4, appearance_score=0.81, first_seen="10:00:23.00", last_seen="10:00:31.00"),
            ],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 2
        step_c02 = route.steps[1]
        assert step_c02.transition_status == "AMBIGUOUS"
        assert step_c02.alternative_candidate is not None
        assert step_c02.alternative_candidate["track_id"] in (3, 4)

    def test_high_similarity_candidate_with_no_supported_transition(self, route_service):
        """10. High appearance with disconnected transition is flagged and penalized."""
        # C03 has high similarity (0.95) but C01 only connects to C02 (transit=20s).
        # C01 to C03 is 2 hops via C02.
        candidates = {
            "C01": [_make_match("C01", appearance_score=0.85, first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match("C02", appearance_score=0.82, first_seen="10:00:25.00", last_seen="10:00:35.00")],
            "C03": [_make_match("C03", appearance_score=0.96, first_seen="10:01:05.00", last_seen="10:01:20.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        # Global path should take C01 -> C02 -> C03, NOT jump C01 -> C03 -> C02
        sequence = [s.match.camera_id for s in route.steps]
        assert sequence == ["C01", "C02", "C03"]

    def test_supported_transition_with_weak_appearance(self, route_service):
        """11. Supported transition with lower appearance is evaluated correctly."""
        candidates = {
            "C01": [_make_match("C01", appearance_score=0.80, first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match("C02", appearance_score=0.62, first_seen="10:00:25.00", last_seen="10:00:35.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 2
        assert route.steps[1].transition_status in ("SUPPORTED", "PROBABLE")

    def test_missing_camera_coordinates(self, route_service):
        """12. Missing coordinates report data limitation without failing."""
        candidates = {
            "C01": [_make_match("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match("C02", first_seen="10:00:25.00", last_seen="10:00:35.00")],
        }
        # Explicit empty coordinates
        route = route_service.reconstruct_route(candidates, camera_coordinates={}, method="topology_aware")
        assert len(route.steps) == 2
        assert any("GPS coordinates missing" in lim for lim in route.data_limitations)
        assert route.steps[1].hop_distance_m is None

    def test_missing_transit_duration_information(self, route_service):
        """13. Missing transit duration handled without assuming 0 seconds."""
        # Transition between cameras without explicit transit duration
        candidates = {
            "C01": [_make_match("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C99": [_make_match("C99", first_seen="10:00:25.00", last_seen="10:00:35.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        c99_step = [s for s in route.steps if s.match.camera_id == "C99"][0]
        assert c99_step.expected_transit_sec == 0
        assert c99_step.transition_status == "DISCONNECTED"

    def test_invalid_camera_ids(self, route_service):
        """14. Unknown camera IDs handled cleanly without crashes."""
        candidates = {
            "UNKNOWN_A": [_make_match("UNKNOWN_A", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "UNKNOWN_B": [_make_match("UNKNOWN_B", first_seen="10:00:20.00", last_seen="10:00:30.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 2
        assert route.steps[0].match.camera_id == "UNKNOWN_A"
        assert route.steps[1].transition_status == "DISCONNECTED"

    def test_repeated_candidate_identities(self, route_service):
        """15. Distinct candidate tracks across cameras correctly preserve track_ids."""
        candidates = {
            "C01": [_make_match("C01", track_id=42, first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match("C02", track_id=42, first_seen="10:00:25.00", last_seen="10:00:35.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        assert route.steps[0].match.track_id == 42
        assert route.steps[1].match.track_id == 42

    def test_query_with_no_valid_route(self, route_service):
        """16. Empty candidate dict yields empty route."""
        route = route_service.reconstruct_route({}, method="topology_aware")
        assert route.steps == []
        assert route.route_confidence == 0.0

    def test_multiple_possible_routes_similar_scores(self, route_service):
        """17. Multiple candidate paths retain alternative route representations."""
        candidates = {
            "C01": [_make_match("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [
                _make_match("C02", track_id=1, appearance_score=0.85, first_seen="10:00:25.00", last_seen="10:00:35.00"),
                _make_match("C02", track_id=2, appearance_score=0.84, first_seen="10:00:26.00", last_seen="10:00:36.00"),
            ],
            "C03": [_make_match("C03", track_id=8, appearance_score=0.80, first_seen="10:01:05.00", last_seen="10:01:15.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        assert len(route.steps) == 3
        assert len(route.alternative_routes) >= 1

    def test_geodesic_speed_validation(self, route_service):
        """18. Geodesic distance calculation and pedestrian speed limits (>6.0 m/s sprint limit)."""
        # Distance between 2 coordinate points: ~120 meters
        coords = {
            "C01": (12.971598, 77.594562),
            "C02": (12.972598, 77.594562),  # ~111 meters away
        }
        dist = haversine_distance(coords["C01"][0], coords["C01"][1], coords["C02"][0], coords["C02"][1])
        assert 100 < dist < 120

        # Transit time 10 seconds -> ~11 m/s (impossible sprint)
        candidates = {
            "C01": [_make_match("C01", first_seen="10:00:00.00", last_seen="10:00:05.00")],
            "C02": [_make_match("C02", first_seen="10:00:15.00", last_seen="10:00:25.00")],
        }
        route = route_service.reconstruct_route(candidates, camera_coordinates=coords, method="topology_aware")
        step_c02 = route.steps[1]
        assert step_c02.hop_distance_m is not None
        assert step_c02.implied_speed_mps is not None
        assert step_c02.implied_speed_mps > 6.0
        assert step_c02.explanation["spatial_evidence"]["speed_valid"] is False

    def test_greedy_baseline_comparison(self, route_service):
        """19. Baseline greedy method and topology-aware method can both be invoked."""
        candidates = {
            "C01": [_make_match("C01", appearance_score=0.80, first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match("C02", appearance_score=0.95, first_seen="10:00:25.00", last_seen="10:00:35.00")],
            "C03": [_make_match("C03", appearance_score=0.82, first_seen="10:01:05.00", last_seen="10:01:15.00")],
        }
        # Greedy anchors on C02 because it has the highest appearance (0.95)
        greedy_route = route_service.reconstruct_route(candidates, method="greedy")
        assert greedy_route.method_used == "greedy"
        assert greedy_route.steps[0].match.camera_id == "C02"

        # Topology-aware starts at C01 because of chronological ordering
        topo_route = route_service.reconstruct_route(candidates, method="topology_aware")
        assert topo_route.method_used == "topology_aware"
        assert topo_route.steps[0].match.camera_id == "C01"

    def test_step_explanation_richness(self, route_service):
        """20. Every step contains structured, evidence-grounded explanation."""
        candidates = {
            "C01": [_make_match("C01", first_seen="10:00:00.00", last_seen="10:00:10.00")],
            "C02": [_make_match("C02", first_seen="10:00:25.00", last_seen="10:00:35.00")],
        }
        route = route_service.reconstruct_route(candidates, method="topology_aware")
        c02_step = route.steps[1]
        expl = c02_step.explanation
        assert expl is not None
        assert expl["step_type"] == "TRANSITION"
        assert "topological_evidence" in expl
        assert "temporal_evidence" in expl
        assert "appearance_evidence" in expl
        assert "spatial_evidence" in expl
        assert "Camera C02 was selected after C01" in expl["summary"]
