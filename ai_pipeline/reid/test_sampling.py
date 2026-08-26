"""
test_sampling.py — Phase 3.1
Tests for the Re-ID crop sampling strategy.

Run with:
    python ai_pipeline/reid/test_sampling.py

Covers:
  1. Track with 101 frames, K=5  — evenly spaced, includes first and last.
  2. Track with fewer than K frames  — returns all available.
  3. Track with exactly K frames  — returns all.
  4. Multiple track_ids handled independently.
  5. No duplicate frame numbers within a track.
  6. Determinism — same input always produces the same output.
  7. K is read from config.yaml, not hardcoded.
"""

import sys
from pathlib import Path

# Allow importing from sibling packages
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.sampling import sample_track_frames, load_crops_per_track, CONFIG_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_records(track_id: int, frames: list[int], camera_id: str = "C01") -> list[dict]:
    """Build minimal mock track records for a single track_id."""
    return [
        {
            "camera_id":  camera_id,
            "frame":      f,
            "timestamp":  f"10:00:{f:05.2f}",
            "track_id":   track_id,
            "bbox":       [100.0, 200.0, 300.0, 400.0],
            "confidence": 0.90,
        }
        for f in frames
    ]


def assert_equal(label: str, actual, expected) -> None:
    if actual != expected:
        print(f"  FAIL  {label}: expected {expected}, got {actual}")
        sys.exit(1)
    print(f"  PASS  {label}")


def assert_true(label: str, condition: bool) -> None:
    if not condition:
        print(f"  FAIL  {label}")
        sys.exit(1)
    print(f"  PASS  {label}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_101_frames_k5():
    print("\n[TEST 1] 101 frames, K=5")
    records = make_records(1, list(range(100, 201)))   # frames 100..200
    result  = sample_track_frames(records, 5)
    frames  = [r["frame"] for r in result]
    print("  Selected frames:", frames)

    assert_equal("count",        len(frames), 5)
    assert_equal("first frame",  frames[0],   100)
    assert_equal("last frame",   frames[-1],  200)
    assert_true ("no duplicates", len(frames) == len(set(frames)))
    # Middle three should be approximately 125, 150, 175
    assert_true ("frame 125 or close", any(120 <= f <= 130 for f in frames))
    assert_true ("frame 150 or close", any(145 <= f <= 155 for f in frames))
    assert_true ("frame 175 or close", any(170 <= f <= 180 for f in frames))


def test_fewer_than_k():
    print("\n[TEST 2] 3 frames, K=5 (fewer than K)")
    records = make_records(2, [100, 101, 102])
    result  = sample_track_frames(records, 5)
    frames  = [r["frame"] for r in result]
    print("  Selected frames:", frames)

    assert_equal("count",       len(frames), 3)
    assert_equal("all frames",  frames,      [100, 101, 102])
    assert_true ("no duplicates", len(frames) == len(set(frames)))


def test_exactly_k():
    print("\n[TEST 3] Exactly 5 frames, K=5")
    records = make_records(3, [10, 20, 30, 40, 50])
    result  = sample_track_frames(records, 5)
    frames  = [r["frame"] for r in result]
    print("  Selected frames:", frames)

    assert_equal("count",      len(frames), 5)
    assert_equal("all frames", frames,      [10, 20, 30, 40, 50])
    assert_true ("no duplicates", len(frames) == len(set(frames)))


def test_multiple_tracks():
    print("\n[TEST 4] Multiple track IDs (tracks 10, 20, 30)")
    records = (
        make_records(10, list(range(0, 101)))    # 101 frames
        + make_records(20, [5, 6, 7])            # 3 frames
        + make_records(30, list(range(200, 401))) # 201 frames
    )
    result = sample_track_frames(records, 5)

    by_track: dict[int, list[int]] = {}
    for r in result:
        by_track.setdefault(r["track_id"], []).append(r["frame"])

    print("  Track 10 frames:", by_track[10])
    print("  Track 20 frames:", by_track[20])
    print("  Track 30 frames:", by_track[30])

    assert_equal("track 10 count", len(by_track[10]), 5)
    assert_equal("track 20 count", len(by_track[20]), 3)  # fewer than K
    assert_equal("track 30 count", len(by_track[30]), 5)
    assert_equal("track 10 first", by_track[10][0],   0)
    assert_equal("track 10 last",  by_track[10][-1],  100)
    assert_equal("track 30 first", by_track[30][0],   200)
    assert_equal("track 30 last",  by_track[30][-1],  400)

    for tid, frames in by_track.items():
        assert_true(f"track {tid} no duplicates", len(frames) == len(set(frames)))


def test_no_duplicates_dense():
    print("\n[TEST 5] No duplicate frames — dense track, K=5")
    records = make_records(5, list(range(0, 10)))   # 10 frames
    result  = sample_track_frames(records, 5)
    frames  = [r["frame"] for r in result]
    print("  Selected frames:", frames)

    assert_equal("count",         len(frames), 5)
    assert_true ("no duplicates", len(frames) == len(set(frames)))


def test_determinism():
    print("\n[TEST 6] Determinism — same input produces same output")
    records = make_records(6, list(range(50, 200)))
    result1 = [r["frame"] for r in sample_track_frames(records, 5)]
    result2 = [r["frame"] for r in sample_track_frames(records, 5)]
    print("  Run 1:", result1)
    print("  Run 2:", result2)
    assert_equal("identical outputs", result1, result2)


def test_k_from_config():
    print("\n[TEST 7] crops_per_track loaded from config.yaml")
    k = load_crops_per_track(CONFIG_PATH)
    print(f"  config.yaml reid.crops_per_track = {k}")
    assert_true("k is int",    isinstance(k, int))
    assert_true("k >= 1",      k >= 1)
    assert_equal("k == 5",     k, 5)   # matches what we set in Phase 3.1


def test_real_tracks_json():
    print("\n[TEST 8] Real tracks_C01.json — smoke test")
    import json
    tracks_path = Path(__file__).resolve().parents[2] / "dataset" / "tracks_C01.json"
    if not tracks_path.exists():
        print("  SKIP  tracks_C01.json not found")
        return

    with open(tracks_path) as f:
        all_records = json.load(f)

    k      = load_crops_per_track(CONFIG_PATH)
    result = sample_track_frames(all_records, k)

    unique_tids = len({r["track_id"] for r in all_records})
    sampled_by_track: dict[int, list] = {}
    for r in result:
        sampled_by_track.setdefault(r["track_id"], []).append(r["frame"])

    print(f"  Total input records  : {len(all_records)}")
    print(f"  Unique track IDs     : {unique_tids}")
    print(f"  K (crops_per_track)  : {k}")
    print(f"  Total sampled records: {len(result)}")
    print(f"  Expected max records : {unique_tids * k}")

    assert_true("sampled <= total",       len(result) <= len(all_records))
    assert_true("sampled <= unique*k",    len(result) <= unique_tids * k)
    for tid, frames in sampled_by_track.items():
        assert_true(f"track {tid} no duplicates", len(frames) == len(set(frames)))

    # Show top-5 longest tracks and their selected frames
    track_lengths = {tid: len(frames) for tid, frames in sampled_by_track.items()}
    top5 = sorted(track_lengths, key=lambda t: -len({r["frame"]
           for r in all_records if r["track_id"] == t}))[:5]
    print("  Sample from 5 longest tracks:")
    for tid in top5:
        print(f"    track_id={tid:>3}  selected frames: {sampled_by_track[tid]}")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_101_frames_k5()
    test_fewer_than_k()
    test_exactly_k()
    test_multiple_tracks()
    test_no_duplicates_dense()
    test_determinism()
    test_k_from_config()
    test_real_tracks_json()

    print("\n=== All tests passed. ===")
