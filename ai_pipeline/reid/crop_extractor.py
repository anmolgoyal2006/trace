"""
crop_extractor.py — Phase 3.2 / updated Phase 3.5
Extract person crop images from the original video using pre-computed
ByteTrack bounding boxes, sampling frames via Phase 3.1 strategy.

Does NOT run YOLO or ByteTrack. Does NOT generate Re-ID embeddings.
Reads tracks JSON → samples frames (sampling.py) → seeks video →
evaluates crop quality (quality_filter.py) → saves JPEGs.

Phase 3.5 additions
-------------------
* Quality filtering via quality_filter.py (size + severe edge truncation).
* --dry-run mode: reports what would be accepted/rejected without touching
  any files on disk.
* Rejection reasons written to dataset/crop_quality_report.json.

Metadata contract (Phase 3.3 / 3.5):
  crop_path, camera_id, track_id, frame, timestamp, bbox, detection_confidence

Usage (normal):
    python ai_pipeline/reid/crop_extractor.py \\
        --video   dataset/raw_videos/clip_001_C01.mp4 \\
        --tracks  dataset/tracks_C01.json \\
        --output-dir dataset/crops \\
        --metadata   dataset/crops_metadata.json

Usage (dry-run — no files written):
    python ai_pipeline/reid/crop_extractor.py \\
        --video   dataset/raw_videos/clip_001_C01.mp4 \\
        --tracks  dataset/tracks_C01.json \\
        --output-dir dataset/crops \\
        --metadata   dataset/crops_metadata.json \\
        --dry-run
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

# Reuse Phase 3.1 sampling — no duplication of algorithm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampling import sample_track_frames          # noqa: E402
from quality_filter import evaluate_crop_quality, load_quality_config  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

# Default location for the quality report (can be overridden via --quality-report)
DEFAULT_QUALITY_REPORT = Path("dataset/crop_quality_report.json")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_reid_config(config_path: Path) -> dict:
    """Return the full reid config block from config.yaml."""
    if not config_path.exists():
        sys.exit(f"[ERROR] Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "reid" not in cfg:
        sys.exit(f"[ERROR] 'reid' section missing in {config_path}")
    return cfg["reid"]


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------

def clip_bbox(bbox: list, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """
    Safely convert and clip a bbox [x1,y1,x2,y2] to the image dimensions.

    Returns:
        (x1, y1, x2, y2) as ints, clipped, or None if the result is degenerate.
    """
    x1 = max(0, min(int(bbox[0]), img_w))
    y1 = max(0, min(int(bbox[1]), img_h))
    x2 = max(0, min(int(bbox[2]), img_w))
    y2 = max(0, min(int(bbox[3]), img_h))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_name(camera_id: str, track_id: int, frame: int) -> str:
    """Return the canonical filename for a crop."""
    return f"{camera_id}_track{track_id}_frame{frame:04d}.jpg"


# ---------------------------------------------------------------------------
# Dry-run report printer
# ---------------------------------------------------------------------------

def _print_quality_report(
    n_selected: int,
    n_accepted: int,
    rejections: list[dict],
    qcfg: dict,
) -> None:
    """Print the formatted quality report to stdout."""
    n_rejected = len(rejections)
    reason_counts: Counter = Counter(r["reason"] for r in rejections)

    print()
    print("=" * 40)
    print("TRACE Crop Quality Report")
    print("=" * 40)
    print(f"Selected crops : {n_selected}")
    print(f"Accepted       : {n_accepted}")
    print(f"Rejected       : {n_rejected}")
    if reason_counts:
        print("Reasons:")
        for reason, count in sorted(reason_counts.items()):
            print(f"  {reason}: {count}")
    else:
        print("Reasons: (none)")
    print()
    print("Thresholds:")
    print(f"  min_crop_width                : {qcfg['min_crop_width']}")
    print(f"  min_crop_height               : {qcfg['min_crop_height']}")
    print(f"  edge_margin_pixels            : {qcfg['edge_margin_pixels']}")
    print(f"  reject_severe_edge_truncation : {qcfg['reject_severe_edge_truncation']}")
    print("=" * 40)
    print()

    if n_rejected > 0:
        pct = 100.0 * n_rejected / n_selected
        if pct > 30.0:
            print(
                f"[WARNING] {pct:.1f}% of selected crops would be rejected. "
                "Review thresholds — excessive filtering may remove legitimate observations."
            )


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_crops(
    video_path: Path,
    tracks_path: Path,
    output_dir: Path,
    metadata_path: Path,
    quality_report_path: Path,
    dry_run: bool = False,
) -> None:

    reid_cfg = load_reid_config(CONFIG_PATH)
    # quality_filter's own loader resolves the nested quality block
    qcfg = load_quality_config(CONFIG_PATH)

    # --- load tracks ---
    if not tracks_path.exists():
        sys.exit(f"[ERROR] Tracks file not found: {tracks_path}")
    try:
        with open(tracks_path) as f:
            all_records = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] Invalid JSON in {tracks_path}: {e}")

    # --- sample frames via Phase 3.1 ---
    crops_per_track = int(reid_cfg["crops_per_track"])
    sampled         = sample_track_frames(all_records, crops_per_track)
    unique_tracks   = len({r["track_id"] for r in all_records})

    mode_label = "[DRY-RUN]" if dry_run else "[INFO]"

    print(f"{mode_label} Video           : {video_path}")
    print(f"{mode_label} Tracks file     : {tracks_path}")
    print(f"{mode_label} Unique tracks   : {unique_tracks}")
    print(f"{mode_label} Selected frames : {len(sampled)}  (K={crops_per_track} per track)")
    print(f"{mode_label} Min crop size   : {qcfg['min_crop_width']}w × {qcfg['min_crop_height']}h px")
    print(f"{mode_label} Edge truncation : {'enabled' if qcfg['reject_severe_edge_truncation'] else 'disabled'}"
          f" (margin={qcfg['edge_margin_pixels']}px)")
    print(f"{mode_label} Dry-run         : {dry_run}")
    print()

    # --- open video ---
    if not video_path.exists():
        sys.exit(f"[ERROR] Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- extract ---
    saved_metadata: list[dict]  = []
    rejections:     list[dict]  = []
    n_saved   = 0
    n_skipped = 0   # operational skips (frame read errors etc.)
    skip_reasons: dict[str, int] = {}

    # Dimension tracking for report
    accepted_widths:  list[int] = []
    accepted_heights: list[int] = []

    def operational_skip(reason: str, track_id: int, frame: int, extra: str = "") -> None:
        """Non-quality skip: frame out of range, video read failure, etc."""
        nonlocal n_skipped
        n_skipped += 1
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        suffix = f" — {extra}" if extra else ""
        print(f"  [SKIP] track={track_id} frame={frame} — {reason}{suffix}")

    for rec in sampled:
        track_id  = rec["track_id"]
        frame_num = rec["frame"]

        # --- range check ---
        if frame_num < 0 or frame_num >= total_video_frames:
            operational_skip("frame out of range", track_id, frame_num,
                             f"video has {total_video_frames} frames")
            continue

        # --- seek and read ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, img = cap.read()
        if not ret or img is None:
            operational_skip("could not read frame", track_id, frame_num)
            continue

        img_h, img_w = img.shape[:2]

        # --- Phase 3.5 quality evaluation ---
        # evaluate_crop_quality handles bbox validation, clipping, size check
        # and severe truncation in one call.
        result = evaluate_crop_quality(rec["bbox"], img_w, img_h, qcfg)

        if not result["accepted"]:
            # Record rejection for the quality report
            rejections.append({
                "camera_id": rec["camera_id"],
                "track_id":  track_id,
                "frame":     frame_num,
                "timestamp": rec["timestamp"],
                "bbox":      rec["bbox"],
                "reason":    result["reason"],
                "details":   result["details"],
            })
            print(
                f"  [REJECT] track={track_id} frame={frame_num}"
                f" — {result['reason']}: {result['details']}"
            )
            continue

        # --- accepted: track dimensions ---
        # Re-derive clipped dimensions for stats (evaluate_crop_quality already
        # validated the bbox, so these clips are safe).
        x1 = max(0, min(int(rec["bbox"][0]), img_w))
        y1 = max(0, min(int(rec["bbox"][1]), img_h))
        x2 = max(0, min(int(rec["bbox"][2]), img_w))
        y2 = max(0, min(int(rec["bbox"][3]), img_h))
        accepted_widths.append(x2 - x1)
        accepted_heights.append(y2 - y1)

        if dry_run:
            # In dry-run mode: count as accepted but do NOT touch disk
            n_saved += 1
            continue

        # --- crop and save ---
        crop_img = img[y1:y2, x1:x2]
        fname    = crop_name(rec["camera_id"], track_id, frame_num)
        out_path = output_dir / fname
        ok = cv2.imwrite(str(out_path), crop_img)
        if not ok:
            operational_skip("imwrite failed", track_id, frame_num, str(out_path))
            # Remove the width/height we speculatively added
            accepted_widths.pop()
            accepted_heights.pop()
            continue

        n_saved += 1
        saved_metadata.append({
            "crop_path":            str(out_path).replace("\\", "/"),
            "camera_id":            rec["camera_id"],
            "track_id":             track_id,
            "frame":                frame_num,
            "timestamp":            rec["timestamp"],
            "bbox":                 rec["bbox"],
            "detection_confidence": rec["confidence"],
        })

    cap.release()

    n_selected = len(sampled)
    n_rejected = len(rejections)

    # --- print quality report ---
    _print_quality_report(n_selected, n_saved, rejections, qcfg)

    # --- dimension summary ---
    if accepted_widths:
        print(f"[INFO] Accepted crop dimensions:")
        print(f"       width  — min={min(accepted_widths)}  max={max(accepted_widths)}")
        print(f"       height — min={min(accepted_heights)}  max={max(accepted_heights)}")
        print()

    if dry_run:
        print("[DRY-RUN] No files written. Re-run without --dry-run to apply changes.")
        # Still write the quality report so the results are auditable
        _write_quality_report(
            quality_report_path, n_selected, n_saved, n_rejected, rejections, dry_run=True
        )
        return

    # --- write metadata ---
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(saved_metadata, f, indent=2)

    # --- write quality report ---
    _write_quality_report(
        quality_report_path, n_selected, n_saved, n_rejected, rejections, dry_run=False
    )

    # --- final summary ---
    print(f"[INFO] Unique tracks   : {unique_tracks}")
    print(f"[INFO] Selected frames : {n_selected}")
    print(f"[INFO] Crops saved     : {n_saved}")
    print(f"[INFO] Crops rejected  : {n_rejected}")
    if skip_reasons:
        print(f"[INFO] Operational skips: {n_skipped}")
        for reason, count in sorted(skip_reasons.items()):
            print(f"         ↳ {reason}: {count}")
    print(f"[INFO] Output dir      : {output_dir}")
    print(f"[INFO] Metadata written: {metadata_path}")
    print(f"[INFO] Quality report  : {quality_report_path}")


# ---------------------------------------------------------------------------
# Quality report writer
# ---------------------------------------------------------------------------

def _write_quality_report(
    path: Path,
    n_selected: int,
    n_accepted: int,
    n_rejected: int,
    rejections: list[dict],
    dry_run: bool,
) -> None:
    """Write the structured rejection log to JSON."""
    report = {
        "dry_run": dry_run,
        "summary": {
            "selected": n_selected,
            "accepted": n_accepted,
            "rejected": n_rejected,
        },
        "rejections": rejections,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    label = "[DRY-RUN]" if dry_run else "[INFO]"
    print(f"{label} Quality report  : {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRACE — extract Re-ID person crops from video using track bounding boxes."
    )
    parser.add_argument("--video",          required=True,  type=Path)
    parser.add_argument("--tracks",         required=True,  type=Path)
    parser.add_argument("--output-dir",     required=True,  type=Path)
    parser.add_argument("--metadata",       required=True,  type=Path)
    parser.add_argument(
        "--quality-report",
        default=DEFAULT_QUALITY_REPORT,
        type=Path,
        help=f"Path for the JSON quality report (default: {DEFAULT_QUALITY_REPORT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse crops without writing images or modifying metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_crops(
        video_path=args.video,
        tracks_path=args.tracks,
        output_dir=args.output_dir,
        metadata_path=args.metadata,
        quality_report_path=args.quality_report,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
