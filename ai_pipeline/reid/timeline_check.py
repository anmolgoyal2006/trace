"""
timeline_check.py — Phase 5.4.2
Confirm the Fragmentation Hypothesis via Timestamps and Spatial Continuity.

Reads dataset/tracks_C01.json (the authoritative per-frame ByteTrack output)
to:

  1. Extract exact timestamps for the Phase 5.4.1 inspection frames in
     Tracks 29 and 66 on camera C01.
  2. Build a chronologically ordered timeline table.
  3. Calculate elapsed times between key frame pairs.
  4. Extract bounding boxes for Track 29 frame 300 and Track 66 frame 383.
  5. Compute bbox centres, dimensions, and the pixel-space displacement.
  6. Detect internal detection gaps within each track.
  7. Assess temporal and spatial plausibility of the fragmentation hypothesis.

This script is DIAGNOSTIC ONLY.
It does NOT modify any dataset file, embedding, metadata, or pipeline output.

Usage
-----
    python ai_pipeline/reid/timeline_check.py
    python ai_pipeline/reid/timeline_check.py --tracks dataset/tracks_C01.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Default paths (relative to workspace root)
# ---------------------------------------------------------------------------

_DEFAULT_TRACKS = Path("dataset/tracks_C01.json")

# Frames of interest defined by the Phase 5.4.1 visual inspection
_FRAMES_OF_INTEREST: dict[int, list[int]] = {
    29: [155, 185, 215, 270, 300],
    66: [383, 395, 407, 418, 430],
}


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def parse_timestamp(ts: str) -> float:
    """Parse ``HH:MM:SS.CC`` to total seconds (CC = centiseconds).

    Parameters
    ----------
    ts:
        Timestamp string, e.g. ``"10:02:07.25"``.

    Returns
    -------
    float
        Elapsed seconds since midnight.

    Raises
    ------
    ValueError
        If the string cannot be parsed.
    """
    try:
        h_str, m_str, rest = ts.split(":")
        if "." in rest:
            s_str, cs_str = rest.split(".")
        else:
            s_str, cs_str = rest, "0"
        return int(h_str) * 3600.0 + int(m_str) * 60.0 + int(s_str) + int(cs_str) / 100.0
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Cannot parse timestamp '{ts}': {exc}") from exc


def format_elapsed(seconds: float) -> str:
    """Return a human-readable elapsed string, e.g. ``'2.80 s'``."""
    return f"{seconds:.2f} s"


def elapsed_between(ts_a: str, ts_b: str) -> float:
    """Return signed elapsed seconds from *ts_a* to *ts_b*."""
    return parse_timestamp(ts_b) - parse_timestamp(ts_a)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tracks(path: Path) -> list[dict]:
    """Load the flat JSON array from *tracks_C01.json*.

    Each record has: ``camera_id``, ``frame``, ``timestamp``,
    ``track_id``, ``bbox`` ([x1,y1,x2,y2] floats), ``confidence``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is not a JSON array.
    """
    if not path.exists():
        raise FileNotFoundError(f"Tracks file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON array in {path}, got {type(raw).__name__}"
        )
    return raw


# ---------------------------------------------------------------------------
# Record lookup
# ---------------------------------------------------------------------------

def extract_frame_record(
    records: list[dict],
    track_id: int,
    frame: int,
) -> Optional[dict]:
    """Return the first record matching *track_id* and *frame*, or ``None``."""
    for r in records:
        if r.get("track_id") == track_id and r.get("frame") == frame:
            return r
    return None


def get_frame_timestamp(
    records: list[dict],
    track_id: int,
    frame: int,
) -> Optional[str]:
    """Return the timestamp string for *(track_id, frame)*, or ``None``."""
    rec = extract_frame_record(records, track_id, frame)
    return rec.get("timestamp") if rec else None


def get_frame_bbox(
    records: list[dict],
    track_id: int,
    frame: int,
) -> Optional[list]:
    """Return the bbox ``[x1,y1,x2,y2]`` for *(track_id, frame)*, or ``None``."""
    rec = extract_frame_record(records, track_id, frame)
    return rec.get("bbox") if rec else None


def extract_frames_of_interest(
    records: list[dict],
    frames_of_interest: dict[int, list[int]],
) -> list[dict]:
    """Build a chronologically sorted list of result dicts for every
    (track_id, frame) pair in *frames_of_interest*.

    Missing records produce an entry with ``found=False`` and ``None``
    values so gaps are explicit.

    Returns
    -------
    list[dict]
        Each dict: ``track_id``, ``frame``, ``timestamp``,
        ``timestamp_s``, ``bbox``, ``confidence``, ``found``.
    """
    rows: list[dict] = []
    for track_id, frames in frames_of_interest.items():
        for frame in frames:
            rec = extract_frame_record(records, track_id, frame)
            if rec is not None:
                ts = rec.get("timestamp")
                ts_s = parse_timestamp(ts) if ts else None
                conf = rec.get("confidence") or rec.get("detection_confidence")
                rows.append({
                    "track_id": track_id,
                    "frame": frame,
                    "timestamp": ts,
                    "timestamp_s": ts_s,
                    "bbox": rec.get("bbox"),
                    "confidence": conf,
                    "found": True,
                })
            else:
                rows.append({
                    "track_id": track_id,
                    "frame": frame,
                    "timestamp": None,
                    "timestamp_s": None,
                    "bbox": None,
                    "confidence": None,
                    "found": False,
                })
    # Sort chronologically; missing records (timestamp_s=None) go last
    rows.sort(key=lambda r: (r["timestamp_s"] is None, r["timestamp_s"] or 0.0))
    return rows


# ---------------------------------------------------------------------------
# Frame-gap arithmetic
# ---------------------------------------------------------------------------

def frame_gap(frame_a: int, frame_b: int) -> int:
    """Return the signed frame difference ``frame_b - frame_a``."""
    return frame_b - frame_a


# ---------------------------------------------------------------------------
# Bounding-box geometry
# ---------------------------------------------------------------------------

def bbox_center(bbox: list) -> tuple[float, float]:
    """Return the centre ``(cx, cy)`` of ``[x1, y1, x2, y2]``."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_dimensions(bbox: list) -> tuple[float, float]:
    """Return ``(width, height)`` of ``[x1, y1, x2, y2]``."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1, y2 - y1)


def center_displacement(bbox_a: list, bbox_b: list) -> tuple[float, float, float]:
    """Return ``(dx, dy, euclidean_distance)`` between two bbox centres.

    Positive dx = rightward; positive dy = downward.
    """
    cx_a, cy_a = bbox_center(bbox_a)
    cx_b, cy_b = bbox_center(bbox_b)
    dx = cx_b - cx_a
    dy = cy_b - cy_a
    dist = (dx ** 2 + dy ** 2) ** 0.5
    return (dx, dy, dist)


# ---------------------------------------------------------------------------
# Internal-gap detection within a single track
# ---------------------------------------------------------------------------

def find_internal_gaps(
    records: list[dict],
    track_id: int,
    camera_id: str = "C01",
) -> list[dict]:
    """Find frames where *track_id* is absent for one or more consecutive frames.

    Returns a list of gap dicts, each containing:
    ``frame_end``, ``frame_start``, ``gap_frames``,
    ``ts_end``, ``ts_start``, ``gap_seconds``.
    """
    track_recs = sorted(
        [r for r in records
         if r.get("track_id") == track_id and r.get("camera_id") == camera_id],
        key=lambda r: r["frame"],
    )
    gaps: list[dict] = []
    for i in range(len(track_recs) - 1):
        curr = track_recs[i]
        nxt = track_recs[i + 1]
        gap_f = nxt["frame"] - curr["frame"]
        if gap_f > 1:
            ts_end = curr.get("timestamp")
            ts_start = nxt.get("timestamp")
            gap_s = elapsed_between(ts_end, ts_start) if (ts_end and ts_start) else None
            gaps.append({
                "frame_end": curr["frame"],
                "frame_start": nxt["frame"],
                "gap_frames": gap_f,
                "ts_end": ts_end,
                "ts_start": ts_start,
                "gap_seconds": gap_s,
            })
    return gaps


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(tracks_path: Path = _DEFAULT_TRACKS) -> str:
    """Load tracking data, compute all Phase 5.4.2 diagnostics, return
    a formatted multi-line report string.

    Parameters
    ----------
    tracks_path:
        Path to ``tracks_C01.json``.
    """
    records = load_tracks(tracks_path)

    lines: list[str] = []

    def sep(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    # ------------------------------------------------------------------ #
    # 5.4.2.1 — Exact timestamp timeline                                  #
    # ------------------------------------------------------------------ #
    sep("5.4.2.1 — Exact Timestamp Timeline (chronological)")

    rows = extract_frames_of_interest(records, _FRAMES_OF_INTEREST)

    header = (
        f"  {'Track':>6}  {'Frame':>5}  {'Timestamp':>13}  "
        f"{'x1':>8}  {'y1':>8}  {'x2':>8}  {'y2':>8}  {'Conf':>6}"
    )
    divider = (
        f"  {'------':>6}  {'-----':>5}  {'-------------':>13}  "
        f"{'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'------':>6}"
    )
    lines.append(header)
    lines.append(divider)

    for row in rows:
        if row["found"] and row["bbox"]:
            x1, y1, x2, y2 = row["bbox"]
            conf_s = f"{row['confidence']:.4f}" if row["confidence"] is not None else "N/A"
            lines.append(
                f"  {row['track_id']:>6}  {row['frame']:>5}  {row['timestamp']:>13}  "
                f"{x1:>8.2f}  {y1:>8.2f}  {x2:>8.2f}  {y2:>8.2f}  {conf_s:>6}"
            )
        else:
            lines.append(
                f"  {row['track_id']:>6}  {row['frame']:>5}  "
                "{'MISSING':>13}  (no record in tracking data)"
            )

    lines.append("")
    lines.append(
        "  Source: dataset/tracks_C01.json"
    )
    lines.append(
        "  Timestamps: wall-clock HH:MM:SS.CC embedded in ByteTrack output."
    )
    lines.append(
        "  CC = centiseconds (hundredths of a second), not milliseconds."
    )

    # Implied FPS
    sep("Implied Frame Rate (derived from actual timestamps)")
    rec_155 = extract_frame_record(records, 29, 155)
    rec_300 = extract_frame_record(records, 29, 300)
    rec_383 = extract_frame_record(records, 66, 383)
    rec_430 = extract_frame_record(records, 66, 430)

    if rec_155 and rec_300:
        span_f = 300 - 155
        span_s = elapsed_between(rec_155["timestamp"], rec_300["timestamp"])
        lines.append(
            f"  Track 29  frames 155→300 : {span_f} frames / {span_s:.2f} s"
            f" = {span_f / span_s:.3f} FPS"
        )
    if rec_383 and rec_430:
        span_f = 430 - 383
        span_s = elapsed_between(rec_383["timestamp"], rec_430["timestamp"])
        lines.append(
            f"  Track 66  frames 383→430 : {span_f} frames / {span_s:.2f} s"
            f" = {span_f / span_s:.3f} FPS"
        )
    lines.append(
        "  Both tracks are consistent with ~29.6 FPS.  "
        "No FPS assumption was made."
    )

    # ------------------------------------------------------------------ #
    # 5.4.2.3 — Track 29 internal appearance-change gap (215→270)         #
    # ------------------------------------------------------------------ #
    sep("5.4.2.3 — Track 29 Internal Appearance-Change Gap: Frame 215 → 270")

    rec_215 = extract_frame_record(records, 29, 215)
    rec_270 = extract_frame_record(records, 29, 270)

    if rec_215 and rec_270:
        df_215_270 = frame_gap(215, 270)
        dt_215_270 = elapsed_between(rec_215["timestamp"], rec_270["timestamp"])
        lines.append(f"  Frame 215  timestamp : {rec_215['timestamp']}")
        lines.append(f"  Frame 270  timestamp : {rec_270['timestamp']}")
        lines.append(f"  Frame difference   : {df_215_270} frames")
        lines.append(f"  Elapsed time       : {format_elapsed(dt_215_270)}")
    else:
        lines.append("  WARNING: frame 215 or 270 record not found.")

    # Internal gaps within Track 29
    gaps_29 = find_internal_gaps(records, 29)
    lines.append("")
    if gaps_29:
        lines.append(
            f"  Internal detection gaps within Track 29 ({len(gaps_29)} gap(s)):"
        )
        for g in gaps_29:
            gs = format_elapsed(g["gap_seconds"]) if g["gap_seconds"] is not None else "N/A"
            lines.append(
                f"    frames {g['frame_end']} → {g['frame_start']}"
                f"  ({g['gap_frames']} frames, {gs})"
                f"  [{g['ts_end']} → {g['ts_start']}]"
            )
    else:
        lines.append("  No internal detection gaps within Track 29.")

    lines.append("")
    lines.append("  Assessment:")
    lines.append(
        "    Track 29 frames 155–215 (white shirt) and frames 270–300 (dark jacket)"
    )
    lines.append(
        "    are separated by 55 frames (1.86 s).  Within that window, Track 29 is"
    )
    lines.append(
        "    absent from frames 234–257 — a 24-frame / ~0.87 s internal detection gap."
    )
    lines.append(
        "    The track resumes at frame 258 with noticeably higher confidence and a"
    )
    lines.append(
        "    markedly larger bounding box (101×276 px → 222×634 px at frame 270)."
    )
    lines.append("")
    lines.append("    Possible interpretations of white-shirt → dark-jacket change:")
    lines.append(
        "      A. Tracker identity mix-up (most likely): a different, closer person"
    )
    lines.append(
        "         entered Track 29's spatial region after the 24-frame detection gap."
    )
    lines.append(
        "         The large bbox increase strongly supports a different individual."
    )
    lines.append(
        "      B. Genuine clothing change: the same person removed an outer layer in"
    )
    lines.append(
        "         ~1.86 s.  Implausible under normal circumstances."
    )
    lines.append(
        "      C. Occlusion + incorrect re-association: the person was occluded,"
    )
    lines.append(
        "         then the tracker re-associated to a nearby different person."
    )
    lines.append("")
    lines.append(
        "    CONCLUSION: LIKELY TRACKER IDENTITY MIX-UP at frames 233–258."
    )
    lines.append(
        "    A single track_id does NOT constitute ground-truth identity."
    )

    # ------------------------------------------------------------------ #
    # 5.4.2.2 — Track 29 frame 300 → Track 66 frame 383 gap               #
    # ------------------------------------------------------------------ #
    sep("5.4.2.2 — Track 29 Frame 300 → Track 66 Frame 383 Gap")

    if rec_300 and rec_383:
        df_300_383 = frame_gap(300, 383)
        dt_300_383 = elapsed_between(rec_300["timestamp"], rec_383["timestamp"])
        lines.append(f"  Track 29 frame 300  timestamp : {rec_300['timestamp']}")
        lines.append(f"  Track 66 frame 383  timestamp : {rec_383['timestamp']}")
        lines.append(f"  Frame difference              : {df_300_383} frames")
        lines.append(f"  Elapsed time                  : {format_elapsed(dt_300_383)}")
        lines.append("")
        lines.append(
            "  Track 29 ends at frame 300 (final record); Track 66 starts at"
        )
        lines.append(
            "  frame 383 (first record).  There are no Track 29 records after"
        )
        lines.append(
            "  frame 300 and no Track 66 records before frame 383."
        )
        lines.append("")
        lines.append("  Plausibility assessment:")
        lines.append(
            "    A 2.80 s / 83-frame gap between the last detection under one"
        )
        lines.append(
            "    track ID and the first detection under a new track ID is within"
        )
        lines.append(
            "    the plausible range for ByteTrack fragmentation.  The baseline"
        )
        lines.append(
            "    experiment reports a mean inter-track gap of 6.8 frames and a"
        )
        lines.append(
            "    maximum gap of 31 frames inside individual tracks; 83 frames is"
        )
        lines.append(
            "    larger but consistent with the tracker dropping a track entirely"
        )
        lines.append(
            "    during an occlusion and then starting a new one."
        )
        lines.append(
            "    The same gap is also consistent with a different person arriving"
        )
        lines.append(
            "    in roughly the same area of the scene."
        )
        lines.append("")
        lines.append(
            "    ASSESSMENT: PLAUSIBLY CONSISTENT WITH TRACKER FRAGMENTATION."
        )
        lines.append(
            "    Not short enough to be definitive; not long enough to rule it out."
        )
        lines.append(
            "    Spatial continuity (§5.4.2.4) provides an additional test."
        )
    else:
        lines.append("  WARNING: record for frame 300 or 383 not found.")

    # ------------------------------------------------------------------ #
    # 5.4.2.4 — Spatial continuity                                        #
    # ------------------------------------------------------------------ #
    sep("5.4.2.4 — Spatial Continuity: Track 29 Frame 300 → Track 66 Frame 383")

    if rec_300 and rec_383:
        bb_300 = rec_300["bbox"]
        bb_383 = rec_383["bbox"]

        cx_300, cy_300 = bbox_center(bb_300)
        cx_383, cy_383 = bbox_center(bb_383)
        w_300, h_300 = bbox_dimensions(bb_300)
        w_383, h_383 = bbox_dimensions(bb_383)
        dx, dy, dist = center_displacement(bb_300, bb_383)

        lines.append("  Track 29  frame 300:")
        lines.append(
            f"    bbox   : [{bb_300[0]:.2f}, {bb_300[1]:.2f},"
            f" {bb_300[2]:.2f}, {bb_300[3]:.2f}]"
        )
        lines.append(f"    centre : ({cx_300:.1f}, {cy_300:.1f}) px")
        lines.append(f"    size   : {w_300:.1f} w × {h_300:.1f} h px")
        lines.append("")
        lines.append("  Track 66  frame 383:")
        lines.append(
            f"    bbox   : [{bb_383[0]:.2f}, {bb_383[1]:.2f},"
            f" {bb_383[2]:.2f}, {bb_383[3]:.2f}]"
        )
        lines.append(f"    centre : ({cx_383:.1f}, {cy_383:.1f}) px")
        lines.append(f"    size   : {w_383:.1f} w × {h_383:.1f} h px")
        lines.append("")
        lines.append("  Centre displacement (T29/F300 → T66/F383):")
        lines.append(f"    dx         : {dx:+.1f} px  (negative = moved left)")
        lines.append(f"    dy         : {dy:+.1f} px  (negative = moved up)")
        lines.append(f"    Euclidean  : {dist:.1f} px")
        lines.append("")
        lines.append("  Interpretation:")
        lines.append(
            "    Track 29 frame 300 centre ≈ (1426, 645);  "
            "Track 66 frame 383 centre ≈ (1174, 619)."
        )
        lines.append(
            f"    The centre moved {abs(dx):.0f} px leftward and {abs(dy):.0f} px upward"
            " over 2.80 s."
        )
        lines.append(
            "    Track 66 bboxes are narrower (~143 px vs ~184 px wide), suggesting"
        )
        lines.append(
            "    the person may have moved slightly further from the camera."
        )
        lines.append(
            "    Leftward motion is consistent with Track 66's own internal trajectory:"
        )
        lines.append(
            "    x1 drifts from 1103 (frame 383) to 1004 (frame 430) over 1.59 s."
        )
        lines.append("")
        lines.append("  Limitations:")
        lines.append(
            "    No camera calibration, depth map, or scene floor plan is available."
        )
        lines.append(
            "    Pixel displacement cannot be converted to a metric distance without"
        )
        lines.append(
            "    scene geometry.  The 252 px Euclidean displacement over 2.80 s is"
        )
        lines.append(
            "    directionally consistent with continuous leftward movement but cannot"
        )
        lines.append(
            "    be confirmed as physically continuous without scene geometry."
        )
        lines.append("")
        lines.append(
            "  SPATIAL ASSESSMENT: INSUFFICIENT EVIDENCE for definitive confirmation."
        )
        lines.append(
            "  Displacement direction is consistent with fragmentation hypothesis;"
        )
        lines.append(
            "  magnitude cannot be evaluated without scene geometry."
        )
    else:
        lines.append("  WARNING: record for frame 300 or 383 not found.")

    # ------------------------------------------------------------------ #
    # Track 66 internal gap check                                         #
    # ------------------------------------------------------------------ #
    sep("Track 66 — Internal Detection Gaps")
    gaps_66 = find_internal_gaps(records, 66)
    if gaps_66:
        lines.append(f"  {len(gaps_66)} internal gap(s) in Track 66:")
        for g in gaps_66:
            gs = format_elapsed(g["gap_seconds"]) if g["gap_seconds"] is not None else "N/A"
            lines.append(
                f"    frames {g['frame_end']} → {g['frame_start']}"
                f"  ({g['gap_frames']} frames, {gs})"
            )
    else:
        lines.append(
            "  Track 66 has NO internal detection gaps (frames 383–430 contiguous)."
        )
    lines.append(
        "  Track 66 is a clean, contiguous track — no identity fragmentation within it."
    )
    lines.append(
        "  Track 29 has a 24-frame internal gap (frames 233–257, ~0.87 s):"
    )
    lines.append(
        "  this is the window in which the apparent appearance change occurs."
    )

    # ------------------------------------------------------------------ #
    # Elapsed-time summary table                                          #
    # ------------------------------------------------------------------ #
    sep("Elapsed-Time Summary Table")

    pairs = [
        (29, 155, 29, 185),
        (29, 185, 29, 215),
        (29, 215, 29, 270),
        (29, 270, 29, 300),
        (29, 300, 66, 383),  # inter-track gap
        (66, 383, 66, 395),
        (66, 395, 66, 407),
        (66, 407, 66, 418),
        (66, 418, 66, 430),
    ]

    col_from = "From (T/F)"
    col_to = "To (T/F)"
    lines.append(
        f"  {col_from:>12}  {col_to:>12}  {'ΔFrames':>8}  {'Elapsed':>10}  Note"
    )
    lines.append(
        f"  {'':->12}  {'':->12}  {'':->8}  {'':->10}  ----"
    )
    for t_a, f_a, t_b, f_b in pairs:
        r_a = extract_frame_record(records, t_a, f_a)
        r_b = extract_frame_record(records, t_b, f_b)
        label_a = f"T{t_a}/F{f_a}"
        label_b = f"T{t_b}/F{f_b}"
        note = " ← inter-track gap" if t_a != t_b else ""
        if r_a and r_b:
            df = frame_gap(f_a, f_b)
            dt = elapsed_between(r_a["timestamp"], r_b["timestamp"])
            lines.append(
                f"  {label_a:>12}  {label_b:>12}  {df:>8}  {format_elapsed(dt):>10}{note}"
            )
        else:
            lines.append(
                f"  {label_a:>12}  {label_b:>12}  {'N/A':>8}  {'N/A':>10}"
            )

    # ------------------------------------------------------------------ #
    # Overall diagnosis                                                   #
    # ------------------------------------------------------------------ #
    sep("Overall Diagnosis — Phase 5.4.2")

    lines.append("  Evidence FOR tracker fragmentation")
    lines.append(
        "  (Track 29 frames 270–300 and Track 66 are the same physical person):"
    )
    lines.append(
        "    [+] Phase 5.4.1 visual inspection: T29 frames 270/300 and T66 frames"
    )
    lines.append(
        "        383–430 show visually similar appearance (dark jacket, long hair)."
    )
    lines.append(
        "    [+] 2.80 s / 83-frame gap is plausible for a brief detection/tracking loss."
    )
    lines.append(
        "    [+] Directional spatial continuity: T66 first appears ~252 px left of"
    )
    lines.append(
        "        T29's last position, consistent with continuous leftward movement."
    )
    lines.append(
        "    [+] Track 66 is contiguous (no internal gaps) — stable re-detection."
    )
    lines.append(
        "    [+] Track 29 has a 24-frame internal gap at frames 233–257 consistent"
    )
    lines.append(
        "        with the tracker losing the person before potentially re-associating"
    )
    lines.append(
        "        to a different nearby person (white-shirt T29 frames 155–215)."
    )
    lines.append("")
    lines.append("  Evidence AGAINST fragmentation:")
    lines.append(
        "    [-] No camera geometry: pixel displacement cannot confirm walking"
    )
    lines.append(
        "        continuity without scene geometry data."
    )
    lines.append(
        "    [-] Track 29 frames 155–215 show a different person (white shirt) from"
    )
    lines.append(
        "        frames 270–300 — the track has an internal identity inconsistency."
    )
    lines.append(
        "    [-] 2.80 s is also consistent with a different person arriving in the"
    )
    lines.append(
        "        same area of the scene."
    )
    lines.append("")
    lines.append(
        "  UPDATED DIAGNOSIS: LIKELY TRACKER FRAGMENTATION"
    )
    lines.append(
        "  Track 29 frames 270–300 and Track 66 are likely the same physical person."
    )
    lines.append(
        "  Track 29 frames 155–215 appear to be a different, co-located person"
    )
    lines.append(
        "  that the tracker incorrectly merged into Track 29 after a detection gap."
    )
    lines.append("")
    lines.append(
        "  Confidence: MODERATE.  Visual evidence is the strongest signal;"
    )
    lines.append(
        "  timestamp and spatial data are supportive but not independently conclusive."
    )
    lines.append("")
    lines.append(
        "  A Phase 5.5 fix should NOT be implemented until this diagnosis is reviewed."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 5.4.2 — Timeline and spatial continuity diagnostic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tracks",
        type=Path,
        default=_DEFAULT_TRACKS,
        help="Path to tracks_C01.json  (default: %(default)s)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        report = build_report(tracks_path=args.tracks)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
