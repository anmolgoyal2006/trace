"""
route_service.py — RouteService
Spatial + temporal fusion and cross-camera route reconstruction.

Cross-camera ranking fix (Phase 5.4):
    reconstruct_route() now accepts a dict of *candidate lists* per camera
    (dict[str, list[CameraMatch]]) rather than a single best match per camera
    (dict[str, Optional[CameraMatch]]).

    During the greedy walk each unvisited camera's full candidate list is
    evaluated against the spatial+temporal context that has built up so far.
    The track that maximises the *fused* score (appearance + spatial + temporal)
    is selected — not necessarily the track with the highest raw appearance
    score.  This prevents a track that happens to score well in isolation from
    being committed to when a slightly lower-appearance track is a much better
    fit given when and where the person was last seen.

Fusion formula:
    fusion = 0.60 × appearance + 0.25 × spatial + 0.15 × temporal
    (weights from config.fusion_* fields)

Temporal score:
    Gaussian centred on the expected avg_transit_sec between cameras.
    sigma = avg_transit_sec / 2
    score = exp(-0.5 × ((gap - mean) / sigma)²), clamped to [0, 1]
    Defaults to 1.0 when timestamps are missing or on the anchor camera.

Spatial score:
    1.0 → camera directly adjacent in the graph (1 hop)
    0.4 → camera reachable in 2 hops
    0.0 → camera not reachable from previous camera

Route algorithm:
    Greedy: anchor on the camera whose best-fused candidate scores highest
    when scored as a first hop (spatial=1.0, temporal=1.0).
    Then iteratively extend: for each unvisited camera, score every candidate
    track against current context, take the camera+track pair with the highest
    fused score.
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
    Spatial plausibility of a sighting at *camera_id* given the previous hop.

    Returns:
        1.0 — first hop (no previous camera)
        1.0 — direct neighbour (1 hop)
        0.4 — reachable in 2 hops
        0.0 — not reachable / same camera
    """
    if prev_camera_id is None:
        return 1.0
    if camera_id == prev_camera_id:
        return 0.0
    if camera_id in adjacency.get(prev_camera_id, {}):
        return 1.0
    reachable = reachable_in_n_hops(adjacency, prev_camera_id, max_hops=2)
    return 0.4 if camera_id in reachable else 0.0


def temporal_score(
    first_seen: Optional[str],
    prev_last_seen: Optional[str],
    expected_transit_sec: int,
) -> float:
    """
    Gaussian temporal plausibility.

    score = exp(-0.5 × ((gap_sec - expected) / sigma)²)
    sigma = expected_transit_sec / 2

    Returns 1.0 when timestamps are missing or on the anchor hop.
    Returns 0.05 when the observed gap is negative (appeared before leaving).
    """
    if first_seen is None or prev_last_seen is None:
        return 1.0
    if expected_transit_sec <= 0:
        return 1.0

    try:
        gap_sec = _timestamp_diff_seconds(first_seen, prev_last_seen)
    except (ValueError, Exception):
        return 1.0

    if gap_sec < 0:
        return 0.05

    mean  = float(expected_transit_sec)
    sigma = mean / 2.0
    score = math.exp(-0.5 * ((gap_sec - mean) / sigma) ** 2)
    return float(max(0.0, min(1.0, score)))


def _timestamp_diff_seconds(ts_later: str, ts_earlier: str) -> float:
    """Parse HH:MM:SS.ff timestamps and return ts_later − ts_earlier in seconds."""
    def _to_sec(ts: str) -> float:
        parts = ts.strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split(".")
        s  = int(s_parts[0])
        cs = int(s_parts[1]) if len(s_parts) > 1 else 0
        return h * 3600 + m * 60 + s + cs / 100.0

    return _to_sec(ts_later) - _to_sec(ts_earlier)


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
        # camera_candidates: {camera_id: [CameraMatch, ...]}  (from search_camera_top_k)
        route = svc.reconstruct_route(camera_candidates)
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
        camera_candidates: dict[str, list[CameraMatch]],
    ) -> list[FusedSighting]:
        """
        Build an ordered route from per-camera candidate track lists.

        For each camera, all candidate tracks are evaluated against the
        current spatial+temporal context.  The track that maximises the
        fused score is selected — not necessarily the highest-appearance one.

        Args:
            camera_candidates: {camera_id: [CameraMatch, ...]}
                All lists must be non-empty (cameras with no confident match
                should be absent from the dict entirely).

        Returns:
            Ordered list of FusedSighting, earliest sighting first.
            Empty list if camera_candidates is empty.
        """
        # Drop cameras with no candidates (shouldn't happen, but defensive)
        valid: dict[str, list[CameraMatch]] = {
            cam_id: matches
            for cam_id, matches in camera_candidates.items()
            if matches
        }

        if not valid:
            logger.info("[RouteService] No confident matches — empty route")
            return []

        if len(valid) == 1:
            cam_id, matches = next(iter(valid.items()))
            best = matches[0]   # already sorted by appearance desc
            return [FusedSighting(
                match=best,
                spatial=1.0,
                temporal=1.0,
                fusion=self._fuse(best.appearance_score, 1.0, 1.0),
            )]

        # ── Phase 1: choose anchor camera ───────────────────────────────
        # Score every camera's best candidate as a first hop (s=1, t=1),
        # then anchor on the one with the highest fused score.
        anchor_cam = max(
            valid.keys(),
            key=lambda c: self._fuse(valid[c][0].appearance_score, 1.0, 1.0),
        )

        route: list[FusedSighting] = []
        visited: set[str] = set()

        anchor_match = valid[anchor_cam][0]
        route.append(FusedSighting(
            match=anchor_match,
            spatial=1.0,
            temporal=1.0,
            fusion=self._fuse(anchor_match.appearance_score, 1.0, 1.0),
        ))
        visited.add(anchor_cam)

        # ── Phase 2: greedy extension ────────────────────────────────────
        # At each step, for every unvisited camera score ALL its candidate
        # tracks against current context; keep the (camera, track) pair with
        # the highest fused score.
        while True:
            prev          = route[-1]
            prev_cam_id   = prev.match.camera_id
            prev_last_seen = prev.match.last_seen

            best_next: Optional[tuple[str, FusedSighting]] = None

            for cam_id, matches in valid.items():
                if cam_id in visited:
                    continue

                expected_sec = self._adjacency.get(prev_cam_id, {}).get(cam_id, 0)
                if expected_sec == 0:
                    expected_sec = self._min_transit_2hop(prev_cam_id, cam_id)

                s_score = spatial_score(cam_id, prev_cam_id, self._adjacency)

                # Score each candidate track for this camera and keep the best
                for match in matches:
                    t_score = temporal_score(
                        match.first_seen, prev_last_seen, expected_sec
                    )
                    f_score = self._fuse(match.appearance_score, s_score, t_score)

                    candidate = FusedSighting(
                        match=match,
                        spatial=s_score,
                        temporal=t_score,
                        fusion=f_score,
                        expected_transit_sec=expected_sec,
                    )

                    if best_next is None or f_score > best_next[1].fusion:
                        best_next = (cam_id, candidate)

            if best_next is None:
                break

            best_cam_id, best_fused = best_next
            route.append(best_fused)
            visited.add(best_cam_id)

        logger.info(
            "[RouteService] Route: "
            + " → ".join(
                f"{s.match.camera_id}(track={s.match.track_id} f={s.fusion:.3f})"
                for s in route
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
        """Minimum cumulative transit time via 2 hops; 0 if unreachable."""
        for intermediate, t1 in self._adjacency.get(from_cam, {}).items():
            t2 = self._adjacency.get(intermediate, {}).get(to_cam, 0)
            if t2 > 0:
                return t1 + t2
        return 0

    def get_adjacency(self) -> dict[str, dict[str, int]]:
        return self._adjacency

    def get_graph(self) -> dict:
        return self._graph
