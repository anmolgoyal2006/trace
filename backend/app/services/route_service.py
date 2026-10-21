"""
route_service.py — RouteService
Spatial + temporal fusion and cross-camera route reconstruction.

Supports:
1. Topology-aware sequence inference (multi-hypothesis beam search,
   temporal feasibility constraints, hard contradiction penalties,
   geodesic velocity validation when coordinates exist, ambiguity detection,
   and rich per-transition explainability).
2. Legacy greedy walk baseline (retained for regression testing and benchmarking).

Fusion weights (from config.settings):
    appearance: 0.60
    spatial:    0.25
    temporal:   0.15
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from backend.app.config import settings
from backend.app.services.matching_service import CameraMatch


# ---------------------------------------------------------------------------
# Geodesic / Spatial math helpers
# ---------------------------------------------------------------------------

def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the great-circle distance between two points on the Earth (in meters).
    """
    r_earth = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(r_earth * c, 2)


# ---------------------------------------------------------------------------
# Camera graph helpers
# ---------------------------------------------------------------------------

def load_camera_graph(graph_path: Path) -> dict:
    """Load camera_graph.json and return the parsed dict."""
    if not graph_path.exists():
        raise FileNotFoundError(f"Camera graph not found: {graph_path}")
    with open(graph_path, encoding="utf-8") as f:
        return json.load(f)


def build_adjacency(
    graph: dict, mvp_only: bool = True
) -> dict[str, dict[str, int]]:
    """
    Build a flat adjacency map: {camera_id: {neighbour_id: avg_transit_sec}}.

    If mvp_only=True, only cameras where mvp=true are included.
    """
    cameras = graph.get("cameras", {})
    adj: dict[str, dict[str, int]] = {}
    for cam_id, cam_data in cameras.items():
        if mvp_only and not cam_data.get("mvp", False):
            continue
        adj[cam_id] = {}
        for neighbour, edge in cam_data.get("connects_to", {}).items():
            if not mvp_only or cameras.get(neighbour, {}).get("mvp", False):
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


def shortest_path_transit(
    adjacency: dict[str, dict[str, int]],
    start: str,
    end: str,
    max_hops: int = 4,
) -> tuple[int, int]:
    """
    Find shortest path (by hop count) from start to end in adjacency.

    Returns:
        (hop_count, cumulative_expected_transit_sec)
        Returns (0, 0) if unreachable or same camera.
    """
    if start == end or start not in adjacency:
        return (0, 0)

    # Check direct 1-hop first
    if end in adjacency[start]:
        return (1, adjacency[start][end])

    # BFS for multi-hop
    queue: list[tuple[str, int, int]] = [(start, 0, 0)]  # (node, hops, cumulative_time)
    visited: set[str] = {start}

    while queue:
        curr, hops, cum_time = queue.pop(0)
        if hops >= max_hops:
            continue
        for nbr, transit in adjacency.get(curr, {}).items():
            if nbr == end:
                return (hops + 1, cum_time + transit)
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, hops + 1, cum_time + transit))

    return (0, 0)


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def spatial_score(
    camera_id: str,
    prev_camera_id: Optional[str],
    adjacency: dict[str, dict[str, int]],
) -> float:
    """
    Legacy spatial plausibility of a sighting at *camera_id* given the previous hop.

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

    mean = float(expected_transit_sec)
    sigma = mean / 2.0
    score = math.exp(-0.5 * ((gap_sec - mean) / sigma) ** 2)
    return float(max(0.0, min(1.0, score)))


def _timestamp_diff_seconds(ts_later: str, ts_earlier: str) -> float:
    """Parse HH:MM:SS.ff timestamps and return ts_later − ts_earlier in seconds."""
    def _to_sec(ts: str) -> float:
        parts = ts.strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        s_parts = parts[2].split(".")
        s = int(s_parts[0])
        cs = int(s_parts[1]) if len(s_parts) > 1 else 0
        return h * 3600 + m * 60 + s + cs / 100.0

    return _to_sec(ts_later) - _to_sec(ts_earlier)


# ---------------------------------------------------------------------------
# Fused sighting and Route entities
# ---------------------------------------------------------------------------

@dataclass
class FusedSighting:
    """A CameraMatch enriched with spatial, temporal, fusion scores, and explanations."""
    match: CameraMatch
    spatial: float
    temporal: float
    fusion: float
    expected_transit_sec: int = 0
    observed_gap_sec: Optional[float] = None   # actual gap from prev last_seen
    transition_status: str = "SUPPORTED"       # SUPPORTED | PROBABLE | AMBIGUOUS | DISCONNECTED
    hop_distance_m: Optional[float] = None
    implied_speed_mps: Optional[float] = None
    explanation: Optional[dict[str, Any]] = None
    alternative_candidate: Optional[dict[str, Any]] = None


@dataclass
class SkippedCamera:
    """A camera that was searched but not included in the final route."""
    camera_id: str
    camera_location: str
    best_score: Optional[float] = None
    match_status: Optional[str] = None


@dataclass
class ReconstructedRoute:
    """Return value of reconstruct_route()."""
    steps: list[FusedSighting] = field(default_factory=list)
    skipped_cameras: list[SkippedCamera] = field(default_factory=list)
    route_confidence: float = 0.0
    supported_transitions_count: int = 0
    ambiguous_transitions_count: int = 0
    disconnected_transitions_count: int = 0
    data_limitations: list[str] = field(default_factory=list)
    method_used: str = "topology_aware"
    alternative_routes: list[list[FusedSighting]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RouteService
# ---------------------------------------------------------------------------

class RouteService:
    """
    Topology-aware route reconstruction and spatial+temporal fusion.

    Provides:
        - reconstruct_route() : defaults to topology-aware beam search
        - reconstruct_route_greedy() : legacy greedy implementation baseline
        - reconstruct_route_topology_aware() : robust sequence inference
    """

    def __init__(
        self,
        graph_path: Optional[Path] = None,
        camera_coordinates: Optional[dict[str, tuple[float, float]]] = None,
    ) -> None:
        target_path = graph_path or settings.camera_graph_path
        graph = load_camera_graph(target_path)
        self._graph = graph
        self._adjacency = build_adjacency(graph, mvp_only=True)
        self._adjacency_all = build_adjacency(graph, mvp_only=False)
        self._w_app = settings.fusion_appearance_weight
        self._w_spa = settings.fusion_spatial_weight
        self._w_tem = settings.fusion_temporal_weight

        # Camera coordinates: {camera_id: (lat, lng)}
        self._coordinates: dict[str, tuple[float, float]] = {}
        if camera_coordinates:
            self._coordinates.update(camera_coordinates)
        else:
            # Check if camera_graph.json contains coordinates
            for cam_id, cam_data in graph.get("cameras", {}).items():
                if "latitude" in cam_data and "longitude" in cam_data:
                    lat = cam_data.get("latitude")
                    lng = cam_data.get("longitude")
                    if lat is not None and lng is not None:
                        self._coordinates[cam_id] = (float(lat), float(lng))

        logger.info(
            f"[RouteService] Camera graph loaded. "
            f"MVP cameras: {list(self._adjacency.keys())}, "
            f"Mapped coordinates: {list(self._coordinates.keys())}"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def reconstruct_route(
        self,
        camera_candidates: dict[str, list[CameraMatch]],
        all_camera_candidates: Optional[dict[str, list[CameraMatch]]] = None,
        camera_coordinates: Optional[dict[str, tuple[float, float]]] = None,
        method: str = "topology_aware",
    ) -> ReconstructedRoute:
        """
        Build an ordered route from per-camera candidate track lists.

        Args:
            camera_candidates: {camera_id: [CameraMatch, ...]}
                Candidate tracks used for routing (e.g. CONFIDENT_MATCH).
            all_camera_candidates: {camera_id: [CameraMatch, ...]}
                Full candidate dict including skipped cameras.
            camera_coordinates: Optional runtime GPS mapping {cam_id: (lat, lng)}.
            method: 'topology_aware' (default) or 'greedy' (legacy baseline).

        Returns:
            ReconstructedRoute containing ordered steps, skipped cameras,
            metrics, and explanations.
        """
        # Update coordinates if provided at runtime
        coords = dict(self._coordinates)
        if camera_coordinates:
            coords.update(camera_coordinates)

        if method == "greedy":
            return self.reconstruct_route_greedy(
                camera_candidates=camera_candidates,
                all_camera_candidates=all_camera_candidates,
            )

        return self.reconstruct_route_topology_aware(
            camera_candidates=camera_candidates,
            all_camera_candidates=all_camera_candidates,
            coordinates=coords,
        )

    # ------------------------------------------------------------------ #
    # Legacy Greedy Reconstruction (Baseline)                            #
    # ------------------------------------------------------------------ #

    def reconstruct_route_greedy(
        self,
        camera_candidates: dict[str, list[CameraMatch]],
        all_camera_candidates: Optional[dict[str, list[CameraMatch]]] = None,
    ) -> ReconstructedRoute:
        """
        Original greedy route reconstruction implementation.
        Retained as an exact baseline for benchmarking and regression testing.
        """
        valid: dict[str, list[CameraMatch]] = {
            cam_id: matches
            for cam_id, matches in camera_candidates.items()
            if matches
        }

        if not valid:
            logger.info("[RouteService] No confident matches — empty route")
            route: list[FusedSighting] = []
        elif len(valid) == 1:
            cam_id, matches = next(iter(valid.items()))
            best = matches[0]
            route = [FusedSighting(
                match=best,
                spatial=1.0,
                temporal=1.0,
                fusion=self._fuse(best.appearance_score, 1.0, 1.0),
                transition_status="SUPPORTED",
            )]
        else:
            # Anchor on best initial candidate
            anchor_cam = max(
                valid.keys(),
                key=lambda c: self._fuse(valid[c][0].appearance_score, 1.0, 1.0),
            )
            route = []
            visited: set[str] = set()
            anchor_match = valid[anchor_cam][0]
            route.append(FusedSighting(
                match=anchor_match,
                spatial=1.0,
                temporal=1.0,
                fusion=self._fuse(anchor_match.appearance_score, 1.0, 1.0),
                transition_status="SUPPORTED",
            ))
            visited.add(anchor_cam)

            # Greedy walk
            while True:
                prev = route[-1]
                prev_cam_id = prev.match.camera_id
                prev_last_seen = prev.match.last_seen
                best_next: Optional[tuple[str, FusedSighting]] = None

                for cam_id, matches in valid.items():
                    if cam_id in visited:
                        continue

                    expected_sec = self._adjacency.get(prev_cam_id, {}).get(cam_id, 0)
                    if expected_sec == 0:
                        expected_sec = self._min_transit_2hop(prev_cam_id, cam_id)

                    s_score = spatial_score(cam_id, prev_cam_id, self._adjacency)

                    for match in matches:
                        t_score = temporal_score(
                            match.first_seen, prev_last_seen, expected_sec
                        )
                        f_score = self._fuse(match.appearance_score, s_score, t_score)

                        if match.first_seen is not None and prev_last_seen is not None:
                            try:
                                obs_gap: Optional[float] = _timestamp_diff_seconds(
                                    match.first_seen, prev_last_seen
                                )
                            except Exception:
                                obs_gap = None
                        else:
                            obs_gap = None

                        candidate = FusedSighting(
                            match=match,
                            spatial=s_score,
                            temporal=t_score,
                            fusion=f_score,
                            expected_transit_sec=expected_sec,
                            observed_gap_sec=obs_gap,
                            transition_status="SUPPORTED" if s_score > 0 else "DISCONNECTED",
                        )

                        if best_next is None or f_score > best_next[1].fusion:
                            best_next = (cam_id, candidate)

                if best_next is None:
                    break

                best_cam_id, best_fused = best_next
                route.append(best_fused)
                visited.add(best_cam_id)

        skipped_list = self._derive_skipped_cameras(route, camera_candidates, all_camera_candidates)
        conf = self.compute_route_confidence(route)

        return ReconstructedRoute(
            steps=route,
            skipped_cameras=skipped_list,
            route_confidence=conf,
            supported_transitions_count=sum(1 for s in route if s.spatial > 0),
            ambiguous_transitions_count=0,
            disconnected_transitions_count=sum(1 for i, s in enumerate(route) if i > 0 and s.spatial == 0.0),
            data_limitations=["Baseline greedy method used; ambiguity and GPS validation not evaluated."],
            method_used="greedy",
        )

    # ------------------------------------------------------------------ #
    # Topology-Aware Route Reconstruction                                #
    # ------------------------------------------------------------------ #

    def reconstruct_route_topology_aware(
        self,
        camera_candidates: dict[str, list[CameraMatch]],
        all_camera_candidates: Optional[dict[str, list[CameraMatch]]] = None,
        coordinates: Optional[dict[str, tuple[float, float]]] = None,
    ) -> ReconstructedRoute:
        """
        Topology-aware, temporally constrained route reconstruction with
        multi-hypothesis beam search, spatial validation, ambiguity detection,
        and transparent explanations.
        """
        coords = coordinates or self._coordinates
        data_limitations: list[str] = []

        valid: dict[str, list[CameraMatch]] = {
            cam_id: matches
            for cam_id, matches in camera_candidates.items()
            if matches
        }

        if not valid:
            logger.info("[RouteService] No confident matches — empty route")
            return ReconstructedRoute(
                steps=[],
                skipped_cameras=self._derive_skipped_cameras([], camera_candidates, all_camera_candidates),
                route_confidence=0.0,
                method_used="topology_aware",
            )

        # Track global data limitations
        cams_without_coords = [c for c in valid.keys() if c not in coords]
        if cams_without_coords:
            data_limitations.append(
                f"Camera GPS coordinates missing for: {', '.join(sorted(cams_without_coords))}. "
                "Geodesic speed and physical distance checks could not be computed for these cameras."
            )

        cams_with_missing_ts = [
            c for c, m_list in valid.items()
            if any(m.first_seen is None for m in m_list)
        ]
        if cams_with_missing_ts:
            data_limitations.append(
                f"Sighting timestamps missing for candidates in: {', '.join(sorted(cams_with_missing_ts))}. "
                "Temporal continuity and transit speed could not be strictly verified."
            )

        if len(valid) == 1:
            cam_id, matches = next(iter(valid.items()))
            best = matches[0]
            explanation = self._generate_step_explanation(
                dest_match=best,
                prev_match=None,
                edge_hops=0,
                expected_transit_sec=0,
                observed_gap_sec=None,
                geo_dist_m=None,
                implied_speed_mps=None,
                topology_score=1.0,
                temporal_score_val=1.0,
                status="SUPPORTED",
                alternative=matches[1] if len(matches) > 1 else None,
            )
            step = FusedSighting(
                match=best,
                spatial=1.0,
                temporal=1.0,
                fusion=self._fuse(best.appearance_score, 1.0, 1.0),
                expected_transit_sec=0,
                observed_gap_sec=None,
                transition_status="SUPPORTED",
                explanation=explanation,
            )
            skipped_list = self._derive_skipped_cameras([step], camera_candidates, all_camera_candidates)
            return ReconstructedRoute(
                steps=[step],
                skipped_cameras=skipped_list,
                route_confidence=round(step.fusion, 4),
                supported_transitions_count=1,
                ambiguous_transitions_count=0,
                disconnected_transitions_count=0,
                data_limitations=data_limitations,
                method_used="topology_aware",
            )

        # Multi-camera sequence search via Beam Search
        best_path, alt_paths = self._beam_search_route(valid, coords)

        # Compute summary metrics
        supported_cnt = 0
        ambiguous_cnt = 0
        disconnected_cnt = 0

        for i, step in enumerate(best_path):
            if step.transition_status == "AMBIGUOUS":
                ambiguous_cnt += 1
            if i > 0:
                if step.transition_status == "DISCONNECTED":
                    disconnected_cnt += 1
                elif step.transition_status in ("SUPPORTED", "PROBABLE"):
                    supported_cnt += 1
            else:
                supported_cnt += 1

        route_conf = self.compute_route_confidence(best_path)
        skipped_list = self._derive_skipped_cameras(best_path, camera_candidates, all_camera_candidates)

        logger.info(
            f"[RouteService:TopologyAware] Best route ({len(best_path)} steps, conf={route_conf:.3f}): "
            + " → ".join(
                f"{s.match.camera_id}(T{s.match.track_id} status={s.transition_status} f={s.fusion:.3f})"
                for s in best_path
            )
        )

        return ReconstructedRoute(
            steps=best_path,
            skipped_cameras=skipped_list,
            route_confidence=route_conf,
            supported_transitions_count=supported_cnt,
            ambiguous_transitions_count=ambiguous_cnt,
            disconnected_transitions_count=disconnected_cnt,
            data_limitations=data_limitations,
            method_used="topology_aware",
            alternative_routes=alt_paths,
        )

    # ------------------------------------------------------------------ #
    # Beam Search Sequence Optimizer                                     #
    # ------------------------------------------------------------------ #

    def _beam_search_route(
        self,
        valid: dict[str, list[CameraMatch]],
        coords: dict[str, tuple[float, float]],
        beam_width: int = 5,
    ) -> tuple[list[FusedSighting], list[list[FusedSighting]]]:
        """
        Search for the most plausible consistent path using beam search.
        Evaluates alternative candidate starts and scores transitions
        incorporating topology, chronological feasibility, spatial constraints,
        and contradiction penalties.
        """
        # Step 1: Candidate anchor starts
        # Prefer starting from candidates that have earlier timestamps if available,
        # or highest appearance score.
        candidate_anchors: list[tuple[str, CameraMatch]] = []
        for cam_id, matches in valid.items():
            for m in matches[:2]:  # check top 2 per camera
                candidate_anchors.append((cam_id, m))

        # Sort candidate anchors: if timestamps present, sort by first_seen; else by appearance
        def _anchor_sort_key(item: tuple[str, CameraMatch]) -> tuple:
            m = item[1]
            if m.first_seen is not None:
                try:
                    return (0, _timestamp_diff_seconds(m.first_seen, "00:00:00"))
                except Exception:
                    pass
            return (1, -m.appearance_score)

        candidate_anchors.sort(key=_anchor_sort_key)

        # Initialize beam
        # Beam element: (cumulative_score, path: list[FusedSighting], visited: set[str])
        beam: list[tuple[float, list[FusedSighting], set[str]]] = []

        for cam_id, match in candidate_anchors[:beam_width]:
            init_fuse = self._fuse(match.appearance_score, 1.0, 1.0)
            explanation = self._generate_step_explanation(
                dest_match=match,
                prev_match=None,
                edge_hops=0,
                expected_transit_sec=0,
                observed_gap_sec=None,
                geo_dist_m=None,
                implied_speed_mps=None,
                topology_score=1.0,
                temporal_score_val=1.0,
                status="SUPPORTED",
            )
            step = FusedSighting(
                match=match,
                spatial=1.0,
                temporal=1.0,
                fusion=init_fuse,
                expected_transit_sec=0,
                observed_gap_sec=None,
                transition_status="SUPPORTED",
                explanation=explanation,
            )
            beam.append((init_fuse, [step], {cam_id}))

        target_len = len(valid)

        # Iteratively extend paths
        for _ in range(target_len - 1):
            next_beam: list[tuple[float, list[FusedSighting], set[str]]] = []

            for cum_score, path, visited in beam:
                prev = path[-1]
                prev_cam_id = prev.match.camera_id

                unvisited = [c for c in valid.keys() if c not in visited]
                if not unvisited:
                    # Path cannot be extended further (e.g. all cameras visited)
                    next_beam.append((cum_score, path, visited))
                    continue

                # Score all candidate extensions
                candidate_transitions: list[tuple[float, FusedSighting, str]] = []

                for cam_id in unvisited:
                    matches = valid[cam_id]
                    for match in matches:
                        eval_res = self._evaluate_transition(
                            from_match=prev.match,
                            to_match=match,
                            coords=coords,
                        )
                        candidate_transitions.append((eval_res["transition_score"], eval_res["step"], cam_id))

                if not candidate_transitions:
                    next_beam.append((cum_score, path, visited))
                    continue

                # Sort candidate transitions descending
                candidate_transitions.sort(key=lambda x: x[0], reverse=True)

                # Check for step-level ambiguity among top choices
                best_t_score, best_step, best_cam = candidate_transitions[0]
                if len(candidate_transitions) > 1:
                    alt_score, alt_step, alt_cam = candidate_transitions[1]
                    if abs(best_t_score - alt_score) <= 0.05:
                        best_step.transition_status = "AMBIGUOUS"
                        best_step.alternative_candidate = {
                            "camera_id": alt_cam,
                            "track_id": alt_step.match.track_id,
                            "appearance_score": alt_step.match.appearance_score,
                            "fusion_score": alt_step.fusion,
                            "first_seen": alt_step.match.first_seen,
                            "transition_score": round(alt_score, 4),
                        }
                        if best_step.explanation:
                            best_step.explanation["status"] = "AMBIGUOUS"
                            best_step.explanation["ambiguity_note"] = (
                                f"Candidate track {alt_step.match.track_id} at {alt_cam} scored within 0.05 "
                                f"(score delta: {abs(best_t_score - alt_score):.4f})."
                            )

                # Add top extensions to next beam
                for t_score, step_obj, next_cam in candidate_transitions[:beam_width]:
                    # Contradiction pruning: skip impossible negative time transitions unless forced
                    if t_score <= -0.2 and len(candidate_transitions) > 1 and candidate_transitions[0][0] > 0.0:
                        continue
                    new_path = list(path) + [step_obj]
                    new_visited = set(visited) | {next_cam}
                    new_cum = cum_score + step_obj.fusion
                    next_beam.append((new_cum, new_path, new_visited))

            if not next_beam:
                break

            # Deduplicate by visited camera sequence to maintain diversity
            next_beam.sort(key=lambda x: (len(x[1]), x[0] / len(x[1])), reverse=True)
            beam = next_beam[:beam_width]

        if not beam:
            return ([], [])

        # Sort paths by length first, then average fusion score
        beam.sort(key=lambda x: (len(x[1]), x[0] / len(x[1])), reverse=True)
        best_path = beam[0][1]
        alt_paths = [b[1] for b in beam[1:]]

        return best_path, alt_paths

    # ------------------------------------------------------------------ #
    # Transition Evaluation & Scoring Engine                             #
    # ------------------------------------------------------------------ #

    def _evaluate_transition(
        self,
        from_match: CameraMatch,
        to_match: CameraMatch,
        coords: dict[str, tuple[float, float]],
    ) -> dict[str, Any]:
        """
        Evaluate and score a transition between from_match and to_match.

        Computes:
            - Topology compatibility (graph edge existence, hop distance, expected transit)
            - Temporal feasibility (chronological delta, sprint/velocity check, missing timestamp handling)
            - Spatial compatibility (geodesic distance via Haversine, implied walking speed)
            - Contradiction penalties (impossible backward time, disconnected transitions)
            - Transparent explanation object
        """
        src_cam = from_match.camera_id
        dst_cam = to_match.camera_id

        # 1. Topology compatibility
        hop_count, expected_transit = shortest_path_transit(self._adjacency, src_cam, dst_cam)
        if hop_count == 0:
            # Check non-MVP adjacency as stretch
            hop_count_all, expected_transit_all = shortest_path_transit(self._adjacency_all, src_cam, dst_cam)
            if hop_count_all > 0:
                hop_count = hop_count_all
                expected_transit = expected_transit_all

        if hop_count == 1:
            topo_score = 1.0
        elif hop_count == 2:
            topo_score = 0.50
        elif hop_count >= 3:
            topo_score = 0.25
        else:
            topo_score = 0.0  # Disconnected

        # 2. Temporal compatibility
        obs_gap: Optional[float] = None
        temp_score = 0.50  # Neutral default for missing timestamps
        has_timestamps = (to_match.first_seen is not None and from_match.last_seen is not None)
        contradiction_penalty = 0.0
        time_verdict = "unverified"

        if has_timestamps:
            try:
                obs_gap = _timestamp_diff_seconds(to_match.first_seen, from_match.last_seen)
            except Exception:
                obs_gap = None

        if obs_gap is not None:
            if obs_gap < 0:
                # Impossible backward time
                temp_score = 0.0
                contradiction_penalty += 0.60
                time_verdict = "backward_in_time"
            elif obs_gap == 0 and src_cam != dst_cam:
                # Simultaneous appearance at two distinct cameras
                temp_score = 0.0
                contradiction_penalty += 0.50
                time_verdict = "simultaneous_overlap"
            else:
                # Positive time gap
                if expected_transit > 0:
                    min_allowed = max(3.0, expected_transit * 0.25)
                    if obs_gap < min_allowed:
                        # Impossible sprint / teleportation
                        temp_score = 0.05
                        contradiction_penalty += 0.30
                        time_verdict = "too_fast"
                    elif obs_gap < expected_transit * 0.5:
                        time_verdict = "early"
                        mean = float(expected_transit)
                        sigma = max(5.0, mean / 2.0)
                        temp_score = math.exp(-0.5 * ((obs_gap - mean) / sigma) ** 2)
                    elif obs_gap > expected_transit * 3.0:
                        time_verdict = "late"
                        mean = float(expected_transit)
                        sigma = max(5.0, mean / 2.0)
                        temp_score = math.exp(-0.5 * ((obs_gap - mean) / sigma) ** 2)
                    else:
                        time_verdict = "on_time"
                        mean = float(expected_transit)
                        sigma = max(5.0, mean / 2.0)
                        temp_score = math.exp(-0.5 * ((obs_gap - mean) / sigma) ** 2)
                else:
                    # Positive gap but no expected transit configured
                    temp_score = 0.70
                    time_verdict = "forward_time_no_graph_transit"

        # 3. Spatial GPS validation (when coordinates exist)
        geo_dist: Optional[float] = None
        implied_speed: Optional[float] = None
        speed_valid: Optional[bool] = None

        if src_cam in coords and dst_cam in coords:
            lat1, lon1 = coords[src_cam]
            lat2, lon2 = coords[dst_cam]
            geo_dist = haversine_distance(lat1, lon1, lat2, lon2)

            if obs_gap is not None and obs_gap > 0:
                implied_speed = round(geo_dist / obs_gap, 2)
                # Pedestrian sprint limit: > 6.0 m/s is implausible for indoor/foot transit
                if implied_speed > 6.0:
                    speed_valid = False
                    contradiction_penalty += 0.40
                else:
                    speed_valid = True

        # Penalize completely disconnected cameras
        if topo_score == 0.0:
            contradiction_penalty += 0.40

        # Compute fusion score (clamped [0, 1])
        app_score = to_match.appearance_score
        fused = self._fuse(app_score, topo_score, temp_score)

        # Transition score for beam search ranking
        # TransitionScore = Appearance + Topology + Temporal - ContradictionPenalties
        trans_score = (
            self._w_app * app_score
            + self._w_spa * topo_score
            + self._w_tem * temp_score
            - contradiction_penalty
        )

        # Determine step status
        if contradiction_penalty > 0.45 or topo_score == 0.0:
            status = "DISCONNECTED"
        elif time_verdict in ("too_fast", "early", "late") or topo_score < 1.0:
            status = "PROBABLE"
        else:
            status = "SUPPORTED"

        # Construct structured explanation
        explanation = self._generate_step_explanation(
            dest_match=to_match,
            prev_match=from_match,
            edge_hops=hop_count,
            expected_transit_sec=expected_transit,
            observed_gap_sec=obs_gap,
            geo_dist_m=geo_dist,
            implied_speed_mps=implied_speed,
            topology_score=topo_score,
            temporal_score_val=temp_score,
            status=status,
            speed_valid=speed_valid,
        )

        step = FusedSighting(
            match=to_match,
            spatial=round(topo_score, 4),
            temporal=round(temp_score, 4),
            fusion=fused,
            expected_transit_sec=expected_transit,
            observed_gap_sec=obs_gap,
            transition_status=status,
            hop_distance_m=geo_dist,
            implied_speed_mps=implied_speed,
            explanation=explanation,
        )

        return {
            "transition_score": trans_score,
            "step": step,
            "status": status,
        }

    # ------------------------------------------------------------------ #
    # Explainability Generation                                          #
    # ------------------------------------------------------------------ #

    def _generate_step_explanation(
        self,
        dest_match: CameraMatch,
        prev_match: Optional[CameraMatch],
        edge_hops: int,
        expected_transit_sec: int,
        observed_gap_sec: Optional[float],
        geo_dist_m: Optional[float],
        implied_speed_mps: Optional[float],
        topology_score: float,
        temporal_score_val: float,
        status: str,
        speed_valid: Optional[bool] = None,
        alternative: Optional[CameraMatch] = None,
    ) -> dict[str, Any]:
        """
        Generate transparent, evidence-grounded explanation for a route step.
        """
        cam_id = dest_match.camera_id
        track_id = dest_match.track_id
        cam_meta = self._graph.get("cameras", {}).get(cam_id, {})
        location = cam_meta.get("location", cam_id)

        if prev_match is None:
            # Anchor hop
            summary = (
                f"Camera {cam_id} ({location}) was selected as the route anchor point. "
                f"Candidate track {track_id} provides the initial observation with appearance similarity "
                f"{dest_match.appearance_score:.3f} and match score {dest_match.best_confidence:.1f}%."
            )
            return {
                "step_type": "ANCHOR",
                "status": "SUPPORTED",
                "summary": summary,
                "camera_id": cam_id,
                "track_id": track_id,
                "appearance_evidence": {
                    "appearance_score": dest_match.appearance_score,
                    "confidence": dest_match.best_confidence,
                    "backbone": dest_match.active_backbone,
                },
                "missing_constraints": [],
            }

        prev_cam = prev_match.camera_id
        prev_loc = self._graph.get("cameras", {}).get(prev_cam, {}).get("location", prev_cam)

        reasons: list[str] = []
        missing: list[str] = []

        # Topology description
        if edge_hops == 1:
            reasons.append(
                f"the camera graph contains a configured direct {prev_cam} \u2192 {cam_id} transition "
                f"(expected transit: {expected_transit_sec}s)"
            )
        elif edge_hops > 1:
            reasons.append(
                f"reachable via {edge_hops} hops from {prev_cam} in camera graph "
                f"(estimated cumulative transit: {expected_transit_sec}s)"
            )
        else:
            reasons.append(
                f"no configured path in camera graph connects {prev_cam} to {cam_id} (topology score: 0.0)"
            )

        # Temporal description
        if observed_gap_sec is not None:
            if observed_gap_sec < 0:
                reasons.append(
                    f"observation timestamp ({dest_match.first_seen}) is chronologically inconsistent "
                    f"with departure from {prev_cam} ({prev_match.last_seen}), indicating negative gap ({observed_gap_sec:.1f}s)"
                )
            else:
                reasons.append(
                    f"observation timestamps are chronologically compatible (observed gap: {observed_gap_sec:.1f}s)"
                )
        else:
            missing.append("Observation timestamps missing on one or both cameras; transit time unverified.")

        # Spatial description
        if geo_dist_m is not None:
            if implied_speed_mps is not None:
                if speed_valid is False:
                    reasons.append(
                        f"geodesic distance is {geo_dist_m:.1f}m with implied speed {implied_speed_mps:.1f} m/s "
                        "(exceeds realistic pedestrian sprint limit)"
                    )
                else:
                    reasons.append(
                        f"geodesic distance is {geo_dist_m:.1f}m with plausible pedestrian speed {implied_speed_mps:.1f} m/s"
                    )
            else:
                reasons.append(f"geodesic distance between cameras is {geo_dist_m:.1f}m")
        else:
            missing.append("Camera GPS coordinates unconfigured; geodesic speed could not be validated.")

        # Appearance
        reasons.append(
            f"appearance match score ({dest_match.best_confidence:.1f}%, sim: {dest_match.appearance_score:.3f}) supports identity"
        )

        full_summary = f"Camera {cam_id} was selected after {prev_cam} because " + "; ".join(reasons) + "."
        if missing:
            full_summary += " Data limitations: " + " ".join(missing)

        explanation = {
            "step_type": "TRANSITION",
            "status": status,
            "from_camera": prev_cam,
            "to_camera": cam_id,
            "summary": full_summary,
            "topological_evidence": {
                "edge_exists": edge_hops > 0,
                "hop_count": edge_hops,
                "expected_transit_sec": expected_transit_sec,
                "topology_score": round(topology_score, 4),
            },
            "temporal_evidence": {
                "has_timestamps": observed_gap_sec is not None,
                "observed_gap_sec": observed_gap_sec,
                "expected_transit_sec": expected_transit_sec,
                "temporal_score": round(temporal_score_val, 4),
            },
            "appearance_evidence": {
                "appearance_score": dest_match.appearance_score,
                "confidence": dest_match.best_confidence,
                "backbone": dest_match.active_backbone,
            },
            "spatial_evidence": {
                "coordinates_available": geo_dist_m is not None,
                "distance_meters": geo_dist_m,
                "implied_speed_mps": implied_speed_mps,
                "speed_valid": speed_valid,
            },
            "missing_constraints": missing,
        }

        if alternative:
            explanation["alternative_considered"] = {
                "camera_id": alternative.camera_id,
                "track_id": alternative.track_id,
                "appearance_score": alternative.appearance_score,
                "rejection_reason": "Lower overall fused transition score or weaker topology support.",
            }

        return explanation

    # ------------------------------------------------------------------ #
    # Helper methods                                                     #
    # ------------------------------------------------------------------ #

    def _fuse(self, appearance: float, spatial: float, temporal: float) -> float:
        """Weighted fusion of appearance, spatial, and temporal scores."""
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

    def _derive_skipped_cameras(
        self,
        route: list[FusedSighting],
        camera_candidates: dict[str, list[CameraMatch]],
        all_camera_candidates: Optional[dict[str, list[CameraMatch]]],
    ) -> list[SkippedCamera]:
        """Derive list of searched cameras that were not selected for the route."""
        all_cands = all_camera_candidates if all_camera_candidates is not None else camera_candidates
        route_cam_ids: set[str] = {f.match.camera_id for f in route}
        cam_meta = self._graph.get("cameras", {})
        skipped_list: list[SkippedCamera] = []

        for cam_id, cand_list in all_cands.items():
            if cam_id in route_cam_ids:
                continue
            if not cand_list:
                continue
            best_match = sorted(cand_list, key=lambda m: getattr(m, "fused_score", m.appearance_score), reverse=True)[0]
            cam_location = cam_meta.get(cam_id, {}).get("location", cam_id)
            skipped_list.append(SkippedCamera(
                camera_id=cam_id,
                camera_location=cam_location,
                best_score=best_match.fused_score,
                match_status=best_match.match_status.value if hasattr(best_match.match_status, "value") else str(best_match.match_status),
            ))

        return skipped_list

    def compute_route_confidence(
        self, route: ReconstructedRoute | list[FusedSighting]
    ) -> float:
        """
        Compute mean fusion score across all route steps.
        Accepts either a ReconstructedRoute or a list of FusedSighting objects.
        """
        steps = route.steps if isinstance(route, ReconstructedRoute) else route
        if not steps:
            return 0.0
        return round(sum(s.fusion for s in steps) / len(steps), 4)

    def get_adjacency(self) -> dict[str, dict[str, int]]:
        return self._adjacency

    def get_graph(self) -> dict:
        return self._graph

    def get_coordinates(self) -> dict[str, tuple[float, float]]:
        return self._coordinates
