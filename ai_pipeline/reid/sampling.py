"""
sampling.py — Phase 3.1
Re-ID crop sampling strategy for TRACE.

For each track_id, select up to K frames evenly spaced across that track's
available frame range, where K = reid.crops_per_track from config.yaml.

This module does NOT extract crops, run YOLO, or touch the Re-ID model.
It only selects which records to use in the later crop extraction step.

Public API:
    sample_track_frames(track_records, crops_per_track) -> list[dict]
    load_crops_per_track(config_path) -> int
"""

import math
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_crops_per_track(config_path: Path = CONFIG_PATH) -> int:
    """
    Read reid.crops_per_track from config.yaml.

    Returns:
        crops_per_track (int) — number of frames to sample per track.

    Raises:
        SystemExit if the config file is missing or the key is absent.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    try:
        k = int(cfg["reid"]["crops_per_track"])
    except (KeyError, TypeError):
        raise KeyError(
            f"'reid.crops_per_track' not found in {config_path}. "
            "Add it under a 'reid:' section."
        )
    if k < 1:
        raise ValueError(f"crops_per_track must be >= 1, got {k}")
    return k


# ---------------------------------------------------------------------------
# Sampling algorithm
# ---------------------------------------------------------------------------

def sample_track_frames(track_records: list[dict], crops_per_track: int) -> list[dict]:
    """
    Select up to crops_per_track records per track_id, evenly spaced across
    each track's available frame range.

    Algorithm:
      1. Group records by track_id.
      2. Sort each group by frame number.
      3. If the track has <= crops_per_track records, keep them all.
      4. Otherwise, pick crops_per_track indices using:
             idx = round(i * (n-1) / (K-1))   for i in 0..K-1
         This always includes the first (i=0) and last (i=K-1) record,
         with the remaining K-2 points evenly distributed between them.
      5. De-duplicate indices (can happen when n < K due to rounding —
         though step 3 already handles that case).

    Args:
        track_records:   Full list of track dicts (all tracks, all frames).
        crops_per_track: Maximum number of frames to select per track (K).

    Returns:
        Flat list of selected record dicts, sorted by (track_id, frame).
        Each dict retains all original fields: camera_id, track_id, frame,
        timestamp, bbox, confidence.
    """
    if crops_per_track < 1:
        raise ValueError(f"crops_per_track must be >= 1, got {crops_per_track}")

    # --- group by track_id ---
    groups: dict[int, list[dict]] = {}
    for rec in track_records:
        tid = rec["track_id"]
        groups.setdefault(tid, []).append(rec)

    selected: list[dict] = []

    for tid in sorted(groups):
        # Sort by frame so indices are stable and deterministic
        frames = sorted(groups[tid], key=lambda r: r["frame"])
        n = len(frames)

        if n <= crops_per_track:
            # Fewer records than K — keep all of them
            chosen = frames
        else:
            # Pick crops_per_track evenly spaced indices over [0, n-1]
            k = crops_per_track
            indices = sorted({
                round(i * (n - 1) / (k - 1))
                for i in range(k)
            })
            chosen = [frames[idx] for idx in indices]

        selected.extend(chosen)

    # Final sort: track_id first, then frame
    selected.sort(key=lambda r: (r["track_id"], r["frame"]))
    return selected
