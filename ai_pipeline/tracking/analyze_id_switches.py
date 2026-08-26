"""
analyze_id_switches.py — Phase 3.4 / Track Investigation
Analyze all track IDs in a tracks JSON file for signs of ID switching.

Heuristics used (none are definitive — all require human review):

  1. SPATIAL_JUMP    — consecutive observations have center distance > threshold
                       relative to bbox diagonal. Large jumps after a gap are
                       weighted by gap size (a 10-frame gap allows more drift).

  2. SIZE_CHANGE     — bbox area changes > threshold between consecutive obs.
                       A sudden 2x area change suggests a different person.

  3. GAP_NEARBY      — at a gap boundary, another track ends/starts within
                       proximity_px of Track X's last/first known position.
                       Potential ID handoff: Track X may have stolen that ID.

  4. SHORT_TRACK     — tracks < min_length frames are likely ghosts or
                       partial detections, not real ID switches.

Outputs a suspicion report sorted by suspicion level (HIGH > MEDIUM > LOW).

Usage:
    python ai_pipeline/tracking/analyze_id_switches.py \
        --tracks dataset/tracks_C01.json \
        --output dataset/tracking_diagnostics/id_switch_report.txt \
        --jump-thresh   0.5   \
        --size-thresh   1.8   \
        --proximity-px  120   \
        --min-length    10
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_tracks(path: Path) -> dict[int, list[dict]]:
    """Return {track_id: [records sorted by frame]}."""
    with open(path) as f:
        data = json.load(f)
    groups: dict[int, list[dict]] = {}
    for rec in data:
        tid = rec["track_id"]
        groups.setdefault(tid, []).append(rec)
    for tid in groups:
        groups[tid].sort(key=lambda r: r["frame"])
    return groups


def bbox_center(rec: dict) -> tuple[float, float]:
    x1, y1, x2, y2 = rec["bbox"]
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_diag(rec: dict) -> float:
    x1, y1, x2, y2 = rec["bbox"]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def bbox_area(rec: dict) -> float:
    x1, y1, x2, y2 = rec["bbox"]
    return max(0.0, (x2 - x1) * (y2 - y1))


def center_dist(r1: dict, r2: dict) -> float:
    cx1, cy1 = bbox_center(r1)
    cx2, cy2 = bbox_center(r2)
    return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Per-track analysis
# ---------------------------------------------------------------------------

def analyze_track(
    tid: int,
    recs: list[dict],
    all_tracks: dict[int, list[dict]],
    frame_index: dict[int, list[dict]],
    jump_thresh: float,
    size_thresh: float,
    proximity_px: float,
    min_length: int,
) -> dict:
    """
    Return a suspicion dict for one track:
      {
        "track_id": int,
        "length": int,
        "frame_span": int,
        "gaps": [...],
        "jump_events": [...],
        "size_events": [...],
        "gap_nearby_events": [...],
        "suspicion": "HIGH" | "MEDIUM" | "LOW" | "GHOST",
        "summary": str,
      }
    """
    result = {
        "track_id": tid,
        "length": len(recs),
        "frame_span": recs[-1]["frame"] - recs[0]["frame"] + 1,
        "first_frame": recs[0]["frame"],
        "last_frame": recs[-1]["frame"],
        "gaps": [],
        "jump_events": [],
        "size_events": [],
        "gap_nearby_events": [],
        "suspicion": "LOW",
        "summary": "",
    }

    if len(recs) < min_length:
        result["suspicion"] = "GHOST"
        result["summary"] = f"Short track ({len(recs)} frames) — likely ghost/edge detection"
        return result

    for i in range(1, len(recs)):
        prev, curr = recs[i - 1], recs[i]
        gap = curr["frame"] - prev["frame"]

        if gap > 1:
            result["gaps"].append({
                "from": prev["frame"], "to": curr["frame"], "gap": gap
            })

        # --- spatial jump heuristic ---
        dist = center_dist(prev, curr)
        diag = max(bbox_diag(prev), bbox_diag(curr), 1.0)
        # Allow proportionally more drift for larger gaps
        allowed_diag_multiples = jump_thresh * max(1, gap ** 0.5)
        if dist > diag * allowed_diag_multiples:
            result["jump_events"].append({
                "frame": curr["frame"],
                "gap": gap,
                "dist_px": round(dist, 1),
                "diag_px": round(diag, 1),
                "ratio": round(dist / diag, 2),
            })

        # --- size change heuristic ---
        a_prev = max(bbox_area(prev), 1.0)
        a_curr = max(bbox_area(curr), 1.0)
        size_ratio = max(a_curr / a_prev, a_prev / a_curr)
        if size_ratio > size_thresh and gap <= 3:
            result["size_events"].append({
                "frame": curr["frame"],
                "gap": gap,
                "area_ratio": round(size_ratio, 2),
                "area_before": round(a_prev),
                "area_after": round(a_curr),
            })

    # --- gap-nearby heuristic ---
    for gap_info in result["gaps"]:
        f_before = gap_info["from"]
        f_after  = gap_info["to"]

        # Check: did another track end just before f_before OR start just after f_after?
        for other_tid, other_recs in all_tracks.items():
            if other_tid == tid:
                continue
            # Other track ends near gap start
            if other_recs[-1]["frame"] in range(f_before - 3, f_before + 4):
                d = center_dist(recs[i - 1], other_recs[-1])
                if d < proximity_px:
                    result["gap_nearby_events"].append({
                        "type": "other_ends_at_gap_start",
                        "gap_frame": f_before,
                        "other_tid": other_tid,
                        "other_last_frame": other_recs[-1]["frame"],
                        "dist_px": round(d, 1),
                    })
            # Other track starts near gap end
            if other_recs[0]["frame"] in range(f_after - 3, f_after + 4):
                d = center_dist(recs[i], other_recs[0])
                if d < proximity_px:
                    result["gap_nearby_events"].append({
                        "type": "other_starts_at_gap_end",
                        "gap_frame": f_after,
                        "other_tid": other_tid,
                        "other_first_frame": other_recs[0]["frame"],
                        "dist_px": round(d, 1),
                    })

    # --- suspicion level ---
    n_jumps   = len(result["jump_events"])
    n_size    = len(result["size_events"])
    n_nearby  = len(result["gap_nearby_events"])
    n_gaps    = len(result["gaps"])

    if n_jumps >= 2 or (n_jumps >= 1 and n_nearby >= 1) or (n_size >= 1 and n_nearby >= 1):
        result["suspicion"] = "HIGH"
    elif n_jumps == 1 or n_size >= 2 or n_nearby >= 2:
        result["suspicion"] = "MEDIUM"
    elif n_gaps > 0 or n_size == 1 or n_nearby == 1:
        result["suspicion"] = "LOW"
    else:
        result["suspicion"] = "CLEAN"

    parts = []
    if n_jumps:   parts.append(f"{n_jumps} spatial jump(s)")
    if n_size:    parts.append(f"{n_size} sudden size change(s)")
    if n_nearby:  parts.append(f"{n_nearby} nearby track(s) at gap boundary")
    if n_gaps:    parts.append(f"{n_gaps} gap(s) totalling {sum(g['gap'] for g in result['gaps'])} frames")
    result["summary"] = ", ".join(parts) if parts else "no anomalies detected"

    return result


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

LEVEL_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "CLEAN": 3, "GHOST": 4}


def format_report(analyses: list[dict], params: dict) -> str:
    lines = []
    lines.append("=" * 64)
    lines.append("TRACE — ByteTrack ID Switch Analysis Report")
    lines.append("=" * 64)
    lines.append(f"Tracks file       : {params['tracks']}")
    lines.append(f"Total track IDs   : {params['total_tracks']}")
    lines.append(f"Total records     : {params['total_records']}")
    lines.append(f"Jump threshold    : {params['jump_thresh']} × bbox diagonal (scaled by sqrt(gap))")
    lines.append(f"Size threshold    : {params['size_thresh']} × area ratio (gap ≤ 3)")
    lines.append(f"Proximity         : {params['proximity_px']} px (gap-nearby heuristic)")
    lines.append(f"Min track length  : {params['min_length']} frames")
    lines.append("")

    by_level: dict[str, list] = {k: [] for k in LEVEL_ORDER}
    for a in analyses:
        by_level[a["suspicion"]].append(a)

    for level in ["HIGH", "MEDIUM", "LOW", "CLEAN", "GHOST"]:
        group = by_level[level]
        lines.append(f"── {level} ({len(group)} tracks) " + "─" * max(0, 50 - len(level) - 10))
        if not group:
            lines.append("  (none)")
        for a in sorted(group, key=lambda x: x["track_id"]):
            lines.append(
                f"  Track {a['track_id']:>4}  "
                f"len={a['length']:>4}  "
                f"span={a['frame_span']:>4}  "
                f"fr {a['first_frame']:>4}–{a['last_frame']:>4}  "
                f"{a['summary']}"
            )
            if level in ("HIGH", "MEDIUM"):
                for je in a["jump_events"]:
                    lines.append(
                        f"              JUMP  frame={je['frame']}  "
                        f"gap={je['gap']}  dist={je['dist_px']}px  "
                        f"ratio={je['ratio']}×diag"
                    )
                for se in a["size_events"]:
                    lines.append(
                        f"              SIZE  frame={se['frame']}  "
                        f"gap={se['gap']}  "
                        f"area_ratio={se['area_ratio']}  "
                        f"{se['area_before']}→{se['area_after']}px²"
                    )
                for ne in a["gap_nearby_events"]:
                    lines.append(
                        f"              NEARBY  {ne['type']}  "
                        f"gap_at_frame={ne['gap_frame']}  "
                        f"other_tid={ne['other_tid']}  "
                        f"dist={ne['dist_px']}px"
                    )
        lines.append("")

    # Summary table
    lines.append("=" * 64)
    lines.append("SUMMARY")
    lines.append("=" * 64)
    counts = {lv: len(by_level[lv]) for lv in LEVEL_ORDER}
    for lv, cnt in counts.items():
        lines.append(f"  {lv:<8}: {cnt}")
    lines.append("")
    lines.append(
        "NOTE: These are heuristic signals, NOT confirmed ID switches.\n"
        "      Each HIGH/MEDIUM track requires human visual inspection.\n"
        "      LOW/CLEAN tracks are unlikely to have genuine switches.\n"
        "      GHOST tracks are too short for meaningful analysis."
    )
    lines.append("=" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TRACE — heuristic ID-switch analysis for ByteTrack output."
    )
    p.add_argument("--tracks",       type=Path,  default=Path("dataset/tracks_C01.json"))
    p.add_argument("--output",       type=Path,  default=Path("dataset/tracking_diagnostics/id_switch_report.txt"))
    p.add_argument("--jump-thresh",  type=float, default=0.5,
                   help="Max allowed center-displacement as multiple of bbox diagonal per frame (default 0.5)")
    p.add_argument("--size-thresh",  type=float, default=1.8,
                   help="Max allowed area ratio change in ≤3 frames (default 1.8)")
    p.add_argument("--proximity-px", type=float, default=120.0,
                   help="Max px distance to flag a gap-nearby track (default 120)")
    p.add_argument("--min-length",   type=int,   default=10,
                   help="Tracks shorter than this are classified GHOST (default 10)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.tracks.exists():
        sys.exit(f"[ERROR] Tracks file not found: {args.tracks}")

    print(f"[INFO] Loading: {args.tracks}")
    all_tracks = load_tracks(args.tracks)
    total_records = sum(len(v) for v in all_tracks.values())
    print(f"[INFO] Tracks: {len(all_tracks)}  Records: {total_records}")

    # Frame index for gap-nearby lookups
    frame_index: dict[int, list[dict]] = defaultdict(list)
    with open(args.tracks) as f:
        raw = json.load(f)
    for rec in raw:
        frame_index[rec["frame"]].append(rec)

    analyses = []
    for tid, recs in all_tracks.items():
        a = analyze_track(
            tid, recs, all_tracks, frame_index,
            jump_thresh=args.jump_thresh,
            size_thresh=args.size_thresh,
            proximity_px=args.proximity_px,
            min_length=args.min_length,
        )
        analyses.append(a)

    analyses.sort(key=lambda x: (LEVEL_ORDER[x["suspicion"]], x["track_id"]))

    params = {
        "tracks":        str(args.tracks),
        "total_tracks":  len(all_tracks),
        "total_records": total_records,
        "jump_thresh":   args.jump_thresh,
        "size_thresh":   args.size_thresh,
        "proximity_px":  args.proximity_px,
        "min_length":    args.min_length,
    }
    report = format_report(analyses, params)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(report)
    print(f"\n[INFO] Report written: {args.output}")


if __name__ == "__main__":
    main()
