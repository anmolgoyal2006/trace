"""
route_service.py — RouteService
Spatial + temporal fusion and cross-camera route reconstruction.

This is what separates Trace from TraceAI.  Given the per-camera appearance
matches from MatchingService, this service:

  1. Loads the camera adjacency graph from camera_graph.json.
  2. Computes a spatial score for each candidate sighting based on graph
     adjacency from the previous hop.
  3. Computes a temporal score using a Gaussian centred on the expected
     transit time between adjacent cameras.
  4. Fuses appearance + spatial + temporal into a single fusion_score.
  5. Runs a greedy graph walk to build the ordered route.

Fusion formula:
    fusion = 0.60 × appearance + 0.25 × spatial + 0.15 × temporal
    (weights from config.fusion_* fields)

Temporal score:
    Uses a Gaussian PDF evaluated at the observed time gap relative to the
    expected avg_transit_sec from the graph edge.
    sigma = avg_transit_sec / 2
    score = exp(-0.5 × ((gap - mean) / sigma)²), clamped to [0, 1]
    Score = 1.0 when gap == avg_transit_sec exactly.
    Score < 0.05 when gap is more than 2× the expected transit time.
    When no previous camera exists (first hop), spatial and temporal
    scores default to 1.0 (no penalty on the anchor camera).

Spatial score:
    1.0 → camera directly adjacent in the graph (1 hop)
    0.4 → camera reachable in 2 hops
    0.0 → camera not reachable from previous camera

Route algorithm:
    Greedy: start from the camera with the best standalone fusion score,
    then iteratively extend the route by picking the highest-fusion
    adjacent camera that hasn't been visited yet.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.app.config import settings
from backend.app.services.matching_service import CameraMatch


# ---------------------------------------------------------------------------
# Camera graph helpers
# ---------------------------------------------------------------------------

def load_camera_graph(graph_path: Path) -> dict:
    """Load camera_graph.json and return the parsed dict."""
    if not graph_path.exists():
        raise FileNotFoundError(f"Camera graph not found: {graph_path}")
    with open(graph_path) as f:
        return json.load(f)


def build_adjacency(graph: dict) -> dict[str, dict[str, int]]:
    """
    Build a flat adjacency map: {camera_id: {neighbour_id: avg_transit_sec}}.

    Only cameras where mvp=true are included.
    """
    cameras = graph.get("cameras", {})
    adj: dict[str, dict[str, int]] = {}
    for cam_id, cam_data in cameras.items():
        if not cam_data.get("mvp", False):
            continue
        adj[cam_id] = {}
        for neighbour, edge in cam_data.get("connects_to", {}).items():
            if cameras.get(neighbour, {}).get("mvp", False):
                adj[cam_id][neighbour] = int(edge.get("avg_transit_sec", 60))
    return adj


def reachable_in_n_hops(
    adjacency: dict[str, dict[str, int]],
    start: str,
    max_hops: int = 2,
) -> dict[str, int]:
    """
    BFS from *start* up to *max_hops* hops.

    Returns {camera_id: hop_count} for all reachable cameras (excluding start).
    """
    visited: dict[str, int] = {}
    frontier = [(start, 0)]
    while frontier:
        node, depth = frontier.pop(0)
        if depth >= max_hops:
            continue
        for neighbour in adjacency.get(node, {}):
            if neighbour not in visited and neighbour != start:
                visited[neighbour] = depth + 1
                frontier.append((neighbour, depth + 1))
    return visited


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def spatial_score(
    camera_id: str,
    prev_camera_id: Optional[str],
    adjacency: dict[str, dict[str, int]],
) -> float:
    """
    Compute spatial plausibility of a sighting at *camera_id* given that
    the previous confirmed sighting was at *prev_camera_id*.

    Returns:
        1.0 — direct neighbour (1 hop)
        0.4 — reachable in 2 hops
        0.0 — not reachable
        1.0 — when prev_camera_id is None (first hop, no penalty)
    """
    if prev_camera_id is None:
        return 1.0
    if camera_id == prev_camera_id:
        return 0.0  # same camera — no route progress
    direct_neighbours = adjacency.get(prev_camera_id, {})
    if camera_id in direct_neighbours:
        return 1.0
    # Check 2-hop reachability
    reachable = reachable_in_n_hops(adjacency, prev_camera_id, max_hops=2)
    if camera_id in reachable:
        return 0.4
    return 0.0


def temporal_score(
    first_seen: Optional[str],
    prev_last_seen: Optional[str],
    expected_transit_sec: int,
) -> float:
    """
    Gaussian temporal plausibility.

    Score = exp(-0.5 × ((gap_sec - expected) / sigma)²)
    sigma = expected_transit_sec / 2  (so 2σ = expected time)

    Returns 1.0 when:
      - either timestamp is missing (can't compute — no penalty)
      - observed gap == expected transit time

    Returns close to 0.0 when the observed gap is far from expected.
    """
    if first_seen is None or prev_last_seen is None:
        return 1.0
    if expected_transit_sec <= 0:
        return 1.0

    try:
        gap_sec = _timestamp_diff_seconds(first_seen, prev_last_seen)
    except (ValueError, Exception):
        return 1.0  # unparseable timestamps — no penalty

    if gap_sec < 0:
        # Person appeared before they left the previous camera — penalise heavily
        return 0.05

    mean = float(expected_transit_sec)
    sigma = mean / 2.0
    score = math.exp(-0.5 * ((gap_sec - mean) / sigma) ** 2)
    return float(max(0.0, min(1.0, score)))


def _timestamp_diff_seconds(ts_later: str, ts_earlier: str) -> float:
    """
    Parse two HH:MM:SS.ff timestamps and return the difference in seconds.

    ts_later − ts_earlier.  Negative if ts_later is before ts_earlier.
    """
    def _to_seconds(ts: str) -> float:
        parts = ts.strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split(".")
        s = int(s_parts[0])
        cs = int(s_parts[1]) if len(s_parts) > 1 else 0
        return h * 3600 + m * 60 + s + cs / 100.0

    return _to_seconds(ts_later) - _to_seconds(ts_earlier)


# ---------------------------------------------------------------------------
# Fused sighting
# ---------------------------------------------------------------------------

@dataclass
class FusedSighting:
    """A CameraMatch enriched with spatial, temporal, and fusion scores."""
    match: CameraMatch
    spatial: float
    temporal: float
    fusion: float
    expected_transit_sec: int = 0


# ---------------------------------------------------------------------------
# RouteService
# ---------------------------------------------------------------------------

class RouteService:
    """
    Spatial + temporal fusion and route reconstruction.

    Usage:
        svc = RouteService()
        route = svc.reconstruct_route(camera_matches)
        # route is an ordered list of FusedSighting
    """

    def __init__(self) -> None:
        graph = load_camera_graph(settings.camera_graph_path)
        self._adjacency = build_adjacency(graph)
        self._graph = graph
        self._w_app = settings.fusion_appearance_weight
        self._w_spa = settings.fusion_spatial_weight
        self._w_tem = settings.fusion_temporal_weight
        logger.info(
            f"[RouteService] Camera graph loaded. "
            f"MVP cameras: {list(self._adjacency.keys())}"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def reconstruct_route(
        self,
        camera_matches: dict[str, Optional[CameraMatch]],
    ) -> list[FusedSighting]:
        """
        Build an ordered route from per-camera appearance matches.

        Args:
            camera_matches: {camera_id: CameraMatch | None}
                            None entries mean no confident match for that camera.

        Returns:
            Ordered list of FusedSighting, earliest sighting first.
            Empty list if no confident matches exist.
        """
        # Filter to cameras that have a confident match
        valid: dict[str, CameraMatch] = {
            cam_id: match
            for cam_id, match in camera_matches.items()
            if match is not None
        }

        if not valid:
            logger.info("[RouteService] No confident matches — empty route")
            return []

        if len(valid) == 1:
            # Single camera match — no route to reconstruct, just wrap and return
            cam_id, match = next(iter(valid.items()))
            return [FusedSighting(
                match=match,
                spatial=1.0,
                temporal=1.0,
                fusion=match.appearance_score,
            )]

        # ── Phase 1: anchor on the chronologically first sighting ────────
        # Sort by first_seen timestamp to establish temporal ordering
        ordered_cameras = sorted(
            valid.keys(),
            key=lambda c: valid[c].first_seen or "99:99:99",
        )

        # ── Phase 2: greedy route extension ──────────────────────────────
        route: list[FusedSighting] = []
        visited: set[str] = set()

        # Start from the temporally first camera
        first_cam = ordered_cameras[0]
        first_match = valid[first_cam]
        first_fused = FusedSighting(
            match=first_match,
            spatial=1.0,           # anchor — no previous camera
            temporal=1.0,          # anchor — no previous timestamp
            fusion=self._fuse(first_match.appearance_score, 1.0, 1.0),
        )
        route.append(first_fused)
        visited.add(first_cam)

        # Extend greedily
        while True:
            prev = route[-1]
            prev_cam_id = prev.match.camera_id
            prev_last_seen = prev.match.last_seen

            # Score all unvisited candidate cameras
            candidates: list[tuple[str, FusedSighting]] = []
            for cam_id, match in valid.items():
                if cam_id in visited:
                    continue

                # Expected transit time from previous camera → this camera
                expected_sec = self._adjacency.get(prev_cam_id, {}).get(cam_id, 0)
                # If not direct neighbour, get min transit via 2-hop
                if expected_sec == 0:
                    expected_sec = self._min_transit_2hop(prev_cam_id, cam_id)

                s_score = spatial_score(cam_id, prev_cam_id, self._adjacency)
                t_score = temporal_score(match.first_seen, prev_last_seen, expected_sec)
                f_score = self._fuse(match.appearance_score, s_score, t_score)

                candidates.append((cam_id, FusedSighting(
                    match=match,
                    spatial=s_score,
                    temporal=t_score,
                    fusion=f_score,
                    expected_transit_sec=expected_sec,
                )))

            if not candidates:
                break  # no more unvisited cameras with matches

            # Pick the candidate with the highest fusion score
            best_cam_id, best_fused = max(candidates, key=lambda x: x[1].fusion)

            route.append(best_fused)
            visited.add(best_cam_id)

        logger.info(
            f"[RouteService] Route: "
            + " → ".join(
                f"{s.match.camera_id}(f={s.fusion:.3f})" for s in route
            )
        )
        return route

    def compute_route_confidence(self, route: list[FusedSighting]) -> float:
        """Mean fusion score across all route steps."""
        if not route:
            return 0.0
        return round(sum(s.fusion for s in route) / len(route), 4)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _fuse(self, appearance: float, spatial: float, temporal: float) -> float:
        score = (
            self._w_app * appearance
            + self._w_spa * spatial
            + self._w_tem * temporal
        )
        return round(float(max(0.0, min(1.0, score))), 6)

    def _min_transit_2hop(self, from_cam: str, to_cam: str) -> int:
        """
        Find the minimum cumulative transit time between two cameras via 2 hops.
        Returns 0 if not reachable within 2 hops.
        """
        for intermediate, t1 in self._adjacency.get(from_cam, {}).items():
            t2 = self._adjacency.get(intermediate, {}).get(to_cam, 0)
            if t2 > 0:
                return t1 + t2
        return 0

    def get_adjacency(self) -> dict[str, dict[str, int]]:
        """Return the adjacency map for use by other services."""
        return self._adjacency

    def get_graph(self) -> dict:
        """Return the full camera graph dict."""
        return self._graph
