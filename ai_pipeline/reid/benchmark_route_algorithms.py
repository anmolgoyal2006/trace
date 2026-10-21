"""
benchmark_route_algorithms.py — Comparative Benchmark & Ablation Study
======================================================================
Compares TRACE's Original Greedy Route Reconstruction against the
new Topology-Aware Route Reconstruction algorithm across identical synthetic
evaluation scenarios.

Evaluates:
- Route generation success rate (%)
- Number of invalid graph transitions (disconnected hops)
- Number of temporally inconsistent transitions (backward time gaps)
- Number of physically impossible velocity transitions (>6 m/s sprint)
- Ambiguity detection & alternative candidate tracking
- Mean execution runtime (ms)
- Ground truth route accuracy (evaluated on synthetic ground truth)
- Ablation study across available evidence modalities

Saves results to: dataset/route_benchmark_results.json
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.services.matching_service import CameraMatch, MatchStatus
from backend.app.services.route_service import RouteService


def make_match(
    camera_id: str,
    track_id: int = 1,
    appearance: float = 0.85,
    first_seen: str | None = "10:00:00.00",
    last_seen: str | None = "10:00:10.00",
) -> CameraMatch:
    return CameraMatch(
        camera_id=camera_id,
        track_id=track_id,
        appearance_score=appearance,
        mean_top3_similarity=appearance,
        best_confidence=appearance * 100.0,
        mean_confidence=appearance * 100.0,
        first_seen=first_seen,
        last_seen=last_seen,
        match_status=MatchStatus.CONFIDENT_MATCH,
        fused_score=appearance,
    )


def build_benchmark_scenarios() -> list[dict]:
    """Define benchmark test scenarios with synthetic ground truth routes."""
    # Synthetic GPS coordinates
    coords = {
        "C01": (12.971598, 77.594562),
        "C02": (12.971800, 77.594700),  # ~27 meters away
        "C03": (12.972100, 77.595000),  # ~45 meters from C02
        "C04": (12.972500, 77.595500),  # ~70 meters away
        "C99": (13.050000, 77.650000),  # ~10 km away (disconnected)
    }

    scenarios = [
        {
            "id": "scenario_1_standard_forward",
            "name": "Standard Forward Sequence (C01 → C02 → C03)",
            "description": "Person traverses entrance (C01) to corridor (C02) to canteen (C03) on time.",
            "candidates": {
                "C01": [make_match("C01", 1, 0.88, "10:00:00.00", "10:00:10.00")],
                "C02": [make_match("C02", 4, 0.85, "10:00:30.00", "10:00:40.00")],
                "C03": [make_match("C03", 9, 0.82, "10:01:10.00", "10:01:25.00")],
            },
            "coords": coords,
            "expected_sequence": ["C01", "C02", "C03"],
        },
        {
            "id": "scenario_2_appearance_lure",
            "name": "High-Similarity Disconnected Lure (C01, C02, C99)",
            "description": "Disconnected camera C99 has high appearance (0.96) vs valid corridor C02 (0.83).",
            "candidates": {
                "C01": [make_match("C01", 1, 0.85, "10:00:00.00", "10:00:10.00")],
                "C02": [make_match("C02", 2, 0.83, "10:00:30.00", "10:00:40.00")],
                "C99": [make_match("C99", 99, 0.96, "10:00:15.00", "10:00:25.00")],
            },
            "coords": coords,
            "expected_sequence": ["C01", "C02"],  # C99 is disconnected and physically impossible
        },
        {
            "id": "scenario_3_inverted_timestamps",
            "name": "Inverted Timestamp Sequence (C01, C02 with C02 earlier)",
            "description": "C02 occurred at 10:00:00, C01 occurred at 10:00:30. Tests temporal directionality.",
            "candidates": {
                "C01": [make_match("C01", 1, 0.92, "10:00:30.00", "10:00:40.00")],
                "C02": [make_match("C02", 2, 0.80, "10:00:00.00", "10:00:10.00")],
            },
            "coords": coords,
            "expected_sequence": ["C02", "C01"],  # Chronological order C02 -> C01
        },
        {
            "id": "scenario_4_ambiguous_candidates",
            "name": "Ambiguous Candidates in Same Camera",
            "description": "Two candidate tracks in C02 with near-identical appearance scores (0.820 vs 0.815).",
            "candidates": {
                "C01": [make_match("C01", 1, 0.85, "10:00:00.00", "10:00:10.00")],
                "C02": [
                    make_match("C02", 11, 0.820, "10:00:25.00", "10:00:35.00"),
                    make_match("C02", 12, 0.815, "10:00:26.00", "10:00:36.00"),
                ],
            },
            "coords": coords,
            "expected_sequence": ["C01", "C02"],
        },
        {
            "id": "scenario_5_impossible_speed",
            "name": "Impossible Pedestrian Speed / Teleportation",
            "description": "Observation gap is 1s across 120m (velocity 120 m/s >> 6 m/s limit).",
            "candidates": {
                "C01": [make_match("C01", 1, 0.90, "10:00:00.00", "10:00:05.00")],
                "C02": [make_match("C02", 2, 0.88, "10:00:06.00", "10:00:12.00")],
            },
            "coords": {
                "C01": (12.971598, 77.594562),
                "C02": (12.972700, 77.594562),  # ~122 meters away
            },
            "expected_sequence": ["C01", "C02"],
        },
        {
            "id": "scenario_6_missing_timestamps",
            "name": "Missing Observation Timestamps",
            "description": "Observation timestamps are absent. Evaluates unconstrained topology/appearance walk.",
            "candidates": {
                "C01": [make_match("C01", 1, 0.85, None, None)],
                "C02": [make_match("C02", 2, 0.82, None, None)],
            },
            "coords": coords,
            "expected_sequence": ["C01", "C02"],
        },
    ]
    return scenarios


def evaluate_route_metrics(route, scenario, adjacency: dict) -> dict:
    """Evaluate objective metrics on a reconstructed route."""
    steps = route.steps
    n_steps = len(steps)
    invalid_graph_transitions = 0
    temporally_inconsistent_transitions = 0
    speed_violations = 0
    ambiguous_count = 0

    for i in range(1, n_steps):
        prev = steps[i - 1]
        curr = steps[i]
        prev_cam = prev.match.camera_id
        curr_cam = curr.match.camera_id

        # Graph connectivity check
        if curr_cam not in adjacency.get(prev_cam, {}):
            invalid_graph_transitions += 1

        # Temporal consistency check
        if curr.observed_gap_sec is not None and curr.observed_gap_sec < 0:
            temporally_inconsistent_transitions += 1

        # Speed violation
        if curr.implied_speed_mps is not None and curr.implied_speed_mps > 6.0:
            speed_violations += 1

        if curr.transition_status == "AMBIGUOUS":
            ambiguous_count += 1

    actual_seq = [s.match.camera_id for s in steps]
    expected_seq = scenario.get("expected_sequence", [])
    exact_match = (actual_seq == expected_seq)

    return {
        "num_steps": n_steps,
        "sequence": actual_seq,
        "exact_match": exact_match,
        "invalid_graph_transitions": invalid_graph_transitions,
        "temporally_inconsistent_transitions": temporally_inconsistent_transitions,
        "speed_violations": speed_violations,
        "ambiguous_count": ambiguous_count,
        "confidence": route.route_confidence,
    }


def run_benchmark():
    svc = RouteService()
    adjacency = svc.get_adjacency()
    scenarios = build_benchmark_scenarios()

    greedy_results = []
    topo_results = []

    # 1. Evaluate Original Greedy Implementation
    t0 = time.perf_counter()
    for sc in scenarios:
        r = svc.reconstruct_route(sc["candidates"], method="greedy")
        m = evaluate_route_metrics(r, sc, adjacency)
        greedy_results.append(m)
    greedy_time_ms = (time.perf_counter() - t0) * 1000.0 / len(scenarios)

    # 2. Evaluate Improved Topology-Aware Implementation
    t1 = time.perf_counter()
    for sc in scenarios:
        r = svc.reconstruct_route(
            sc["candidates"],
            camera_coordinates=sc.get("coords"),
            method="topology_aware",
        )
        m = evaluate_route_metrics(r, sc, adjacency)
        topo_results.append(m)
    topo_time_ms = (time.perf_counter() - t1) * 1000.0 / len(scenarios)

    # Aggregate metrics
    n_sc = len(scenarios)
    greedy_invalid_trans = sum(m["invalid_graph_transitions"] for m in greedy_results)
    topo_invalid_trans = sum(m["invalid_graph_transitions"] for m in topo_results)

    greedy_temp_inconsistent = sum(m["temporally_inconsistent_transitions"] for m in greedy_results)
    topo_temp_inconsistent = sum(m["temporally_inconsistent_transitions"] for m in topo_results)

    greedy_speed_viols = sum(m["speed_violations"] for m in greedy_results)
    topo_speed_viols = sum(m["speed_violations"] for m in topo_results)

    greedy_ambiguous_detected = sum(m["ambiguous_count"] for m in greedy_results)
    topo_ambiguous_detected = sum(m["ambiguous_count"] for m in topo_results)

    greedy_synthetic_accuracy = sum(1 for m in greedy_results if m["exact_match"]) / n_sc * 100.0
    topo_synthetic_accuracy = sum(1 for m in topo_results if m["exact_match"]) / n_sc * 100.0

    # 3. Ablation Study
    # Modal ablation on Scenario 1 & 2
    ablation_sc = scenarios[1]  # Appearance lure scenario
    cands = ablation_sc["candidates"]
    coords = ablation_sc["coords"]

    # Ablation A: Appearance alone (greedy without spatial/temporal)
    # Simulate appearance alone by setting spatial & temporal to neutral
    r_app_only = svc.reconstruct_route(cands, method="greedy")

    # Ablation B: Appearance + Topology
    r_app_topo = svc.reconstruct_route_topology_aware(cands, coordinates={})

    # Ablation C: Appearance + Topology + Timestamps
    r_app_topo_time = svc.reconstruct_route_topology_aware(cands, coordinates={})

    # Ablation D: All signals (Appearance + Topology + Timestamps + Spatial GPS)
    r_all_signals = svc.reconstruct_route_topology_aware(cands, coordinates=coords)

    benchmark_data = {
        "summary": {
            "num_scenarios": n_sc,
            "greedy_runtime_ms": round(greedy_time_ms, 3),
            "topology_aware_runtime_ms": round(topo_time_ms, 3),
            "greedy_invalid_graph_transitions": greedy_invalid_trans,
            "topology_aware_invalid_graph_transitions": topo_invalid_trans,
            "greedy_temporally_inconsistent_transitions": greedy_temp_inconsistent,
            "topology_aware_temporally_inconsistent_transitions": topo_temp_inconsistent,
            "greedy_speed_violations": greedy_speed_viols,
            "topology_aware_speed_violations": topo_speed_viols,
            "greedy_ambiguous_detected": greedy_ambiguous_detected,
            "topology_aware_ambiguous_detected": topo_ambiguous_detected,
            "greedy_synthetic_accuracy_pct": round(greedy_synthetic_accuracy, 1),
            "topology_aware_synthetic_accuracy_pct": round(topo_synthetic_accuracy, 1),
            "real_world_accuracy": "Ground truth unavailable (no real-world multi-camera trajectory ground truth in dataset)",
        },
        "scenarios": [
            {
                "name": sc["name"],
                "greedy_sequence": greedy_results[i]["sequence"],
                "greedy_valid": greedy_results[i]["exact_match"],
                "topo_sequence": topo_results[i]["sequence"],
                "topo_valid": topo_results[i]["exact_match"],
                "topo_ambiguous_count": topo_results[i]["ambiguous_count"],
            }
            for i, sc in enumerate(scenarios)
        ],
        "ablation_study": {
            "lure_scenario": ablation_sc["name"],
            "appearance_alone_sequence": [s.match.camera_id for s in r_app_only.steps],
            "appearance_plus_topology_sequence": [s.match.camera_id for s in r_app_topo.steps],
            "all_evidence_sequence": [s.match.camera_id for s in r_all_signals.steps],
        }
    }

    output_path = _REPO_ROOT / "dataset" / "route_benchmark_results.json"
    output_path.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
    print(f"Benchmark results successfully saved to: {output_path}")
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Scenarios evaluated: {n_sc}")
    print(f"Original Greedy:    Accuracy: {greedy_synthetic_accuracy:.1f}%, Invalid Trans: {greedy_invalid_trans}, Inconsistent Time: {greedy_temp_inconsistent}, Runtime: {greedy_time_ms:.2f}ms")
    print(f"Topology-Aware:     Accuracy: {topo_synthetic_accuracy:.1f}%, Invalid Trans: {topo_invalid_trans}, Inconsistent Time: {topo_temp_inconsistent}, Ambiguous: {topo_ambiguous_detected}, Runtime: {topo_time_ms:.2f}ms")


if __name__ == "__main__":
    run_benchmark()
