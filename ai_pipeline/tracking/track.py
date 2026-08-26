"""
track.py — Phase 2.2
Run YOLOv8 + ByteTrack on a single camera video and write structured track records to JSON.

Reuses timestamp utilities and config loading from Phase 1 (detect.py).
Does NOT modify or overwrite detect.py or detections_*.json.

Usage:
    python ai_pipeline/tracking/track.py \
        --video   dataset/raw_videos/clip_001_C01.mp4 \
        --camera-id C01 \
        --output  dataset/tracks_C01.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Reuse Phase 1 utilities — no duplication
# ---------------------------------------------------------------------------
# Insert the detection package onto the path so we can import from it directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detection"))
from detect import load_config, get_camera_start_time, frame_to_timestamp  # noqa: E402

from ultralytics import YOLO  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def run_tracking(
    video_path: Path,
    camera_id: str,
    output_path: Path,
    det_cfg: dict,
    cameras_cfg: dict,
) -> None:
    # --- validate inputs ---
    if not video_path.exists():
        sys.exit(f"[ERROR] Video file not found: {video_path}")

    if not camera_id.strip():
        sys.exit("[ERROR] camera-id must not be empty.")

    # --- load model ---
    model_name = det_cfg["model"]
    print(f"[INFO] Loading model : {model_name}")
    try:
        model = YOLO(model_name)
    except Exception as e:
        sys.exit(f"[ERROR] Could not load YOLO model '{model_name}': {e}")

    confidence = float(det_cfg["confidence"])
    classes    = list(det_cfg["classes"])

    # --- resolve camera start time ---
    start_time = get_camera_start_time(cameras_cfg, camera_id)
    print(f"[INFO] Camera start  : {start_time.strftime('%H:%M:%S')}")

    # --- open video to read metadata ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        sys.exit(f"[ERROR] Invalid FPS ({fps}) for video: {video_path}")

    print(f"[INFO] Video         : {video_path}")
    print(f"[INFO] Camera        : {camera_id}")
    print(f"[INFO] FPS           : {fps:.2f}")
    print(f"[INFO] Frames        : {total_frames}")
    print(f"[INFO] Conf          : {confidence}  |  Classes: {classes}")
    print("[INFO] Starting tracking...\n")

    # --- run ByteTrack via Ultralytics model.track() ---
    # model.track() returns results one frame at a time when a generator is used.
    # persist=True keeps the tracker state alive across frames.
    tracks       = []
    frame_number = 0
    skipped      = 0   # detections with no track_id assigned

    results_gen = model.track(
        source=str(video_path),
        conf=confidence,
        classes=classes,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,    # generator — avoids loading all frames into memory
        verbose=False,
    )

    for result in results_gen:
        frame_tracks = 0

        if result.boxes is not None and result.boxes.id is not None:
            boxes      = result.boxes.xyxy.tolist()
            track_ids  = result.boxes.id.tolist()
            confs      = result.boxes.conf.tolist()

            for bbox, track_id, conf in zip(boxes, track_ids, confs):
                # Require a valid positive integer track ID
                tid = int(track_id)
                if tid <= 0:
                    skipped += 1
                    continue

                x1, y1, x2, y2 = bbox
                tracks.append({
                    "camera_id":  camera_id,
                    "frame":      frame_number,
                    "timestamp":  frame_to_timestamp(frame_number, fps, start_time),
                    "track_id":   tid,
                    "bbox":       [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "confidence": round(float(conf), 4),
                })
                frame_tracks += 1

        # Progress every 100 frames
        if frame_number % 100 == 0 or frame_number == total_frames - 1:
            print(f"  frame {frame_number:>6} / {total_frames}"
                  f"  |  tracks this frame: {frame_tracks}")

        frame_number += 1

    # --- write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tracks, f, indent=2)

    unique_ids = len({r["track_id"] for r in tracks})

    print(f"\n[INFO] Done.")
    print(f"[INFO] Total track records : {len(tracks)}")
    print(f"[INFO] Unique track IDs    : {unique_ids}")
    if skipped:
        print(f"[INFO] Skipped (no ID)     : {skipped}")
    print(f"[INFO] Output written      : {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRACE — YOLOv8 + ByteTrack person tracking for a single camera video."
    )
    parser.add_argument(
        "--video", required=True, type=Path,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--camera-id", required=True,
        help="Camera identifier, e.g. C01, C02, C03",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path for output JSON (default: dataset/tracks_<camera-id>.json)",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    output = args.output or Path(f"dataset/tracks_{args.camera_id}.json")
    det_cfg, cameras_cfg = load_config(CONFIG_PATH)
    run_tracking(
        video_path=args.video,
        camera_id=args.camera_id,
        output_path=output,
        det_cfg=det_cfg,
        cameras_cfg=cameras_cfg,
    )


if __name__ == "__main__":
    main()
