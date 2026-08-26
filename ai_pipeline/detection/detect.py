"""
detect.py — Phase 1.5
Run YOLO person detection on a single camera video and write detections to JSON.
Timestamps are expressed as real wall-clock time using the camera's configured
start_time from config.yaml.

Usage:
    python ai_pipeline/detection/detect.py \
        --video dataset/raw_videos/clip_001_C01.mp4 \
        --camera-id C01 \
        --output dataset/detections_C01.json
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
DEFAULT_START_TIME = "00:00:00"   # fallback when camera not listed in config


def load_config(path: Path) -> tuple[dict, dict]:
    """Return (detection_cfg, cameras_cfg) from config.yaml."""
    if not path.exists():
        sys.exit(f"[ERROR] Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "detection" not in cfg:
        sys.exit(f"[ERROR] 'detection' key missing in {path}")
    cameras_cfg = cfg.get("cameras", {})
    return cfg["detection"], cameras_cfg


def get_camera_start_time(cameras_cfg: dict, camera_id: str) -> datetime:
    """
    Look up the wall-clock start time for a camera from config.
    Falls back to DEFAULT_START_TIME if the camera isn't listed.
    Accepts HH:MM:SS or HH:MM:SS.ffffff formats.
    """
    raw = cameras_cfg.get(camera_id, {}).get("start_time", DEFAULT_START_TIME)
    try:
        # Parse as a time-of-day string; use an arbitrary base date — only the
        # time component matters for our output.
        return datetime.strptime(raw, "%H:%M:%S")
    except ValueError:
        sys.exit(f"[ERROR] Invalid start_time '{raw}' for camera {camera_id}. "
                 f"Expected HH:MM:SS (e.g. 10:02:00).")


# ---------------------------------------------------------------------------
# Timestamp utility (Phase 1.5)
# ---------------------------------------------------------------------------

def frame_to_timestamp(frame_num: int, fps: float, video_start_time: datetime) -> str:
    """
    Convert a frame number to a human-readable wall-clock timestamp string.

    Args:
        frame_num:        Zero-based frame index.
        fps:              Video frames per second (must be > 0).
        video_start_time: Wall-clock datetime when the first frame was recorded.

    Returns:
        Timestamp string in HH:MM:SS.ff format, e.g. "10:02:04.90".

    Examples:
        frame 0,   fps=29.63, start=10:02:00  →  "10:02:00.00"
        frame 145, fps=29.63, start=10:02:00  →  "10:02:04.90"
        frame 633, fps=29.63, start=10:02:00  →  "10:02:21.36"
    """
    elapsed = timedelta(seconds=frame_num / fps)
    wall_time = video_start_time + elapsed
    # Format: HH:MM:SS.ff  (centiseconds — two decimal places)
    centiseconds = wall_time.microsecond // 10000
    return f"{wall_time.hour:02d}:{wall_time.minute:02d}:{wall_time.second:02d}.{centiseconds:02d}"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def run_detection(
    video_path: Path,
    camera_id: str,
    output_path: Path,
    det_cfg: dict,
    cameras_cfg: dict,
) -> None:
    # --- load model ---
    print(f"[INFO] Loading model: {det_cfg['model']}")
    try:
        model = YOLO(det_cfg["model"])
    except Exception as e:
        sys.exit(f"[ERROR] Could not load YOLO model '{det_cfg['model']}': {e}")

    confidence = float(det_cfg["confidence"])
    classes = list(det_cfg["classes"])

    # --- resolve camera start time ---
    start_time = get_camera_start_time(cameras_cfg, camera_id)
    print(f"[INFO] Camera start time: {start_time.strftime('%H:%M:%S')}")

    # --- open video ---
    if not video_path.exists():
        sys.exit(f"[ERROR] Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        sys.exit(f"[ERROR] Invalid FPS ({fps}) for video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video  : {video_path}")
    print(f"[INFO] Camera : {camera_id}")
    print(f"[INFO] FPS    : {fps:.2f}")
    print(f"[INFO] Frames : {total_frames}")
    print(f"[INFO] Conf   : {confidence}  |  Classes: {classes}")
    print("[INFO] Starting detection...\n")

    # --- process frames ---
    detections = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=confidence,
            classes=classes,
            verbose=False,   # suppress per-frame Ultralytics output
        )

        frame_detections = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "camera_id":  camera_id,
                    "frame":      frame_number,
                    "timestamp":  frame_to_timestamp(frame_number, fps, start_time),
                    "bbox":       [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "confidence": round(float(box.conf[0]), 4),
                })
                frame_detections += 1

        # Print progress every 100 frames
        if frame_number % 100 == 0 or frame_number == total_frames - 1:
            print(f"  frame {frame_number:>6} / {total_frames}  |  detections this frame: {frame_detections}")

        frame_number += 1

    cap.release()

    # --- write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(detections, f, indent=2)

    print(f"\n[INFO] Done.")
    print(f"[INFO] Total detections : {len(detections)}")
    print(f"[INFO] Output written   : {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRACE — YOLO person detection for a single camera video."
    )
    parser.add_argument(
        "--video", required=True, type=Path,
        help="Path to input video file (e.g. dataset/raw_videos/clip_001_C01.mp4)",
    )
    parser.add_argument(
        "--camera-id", required=True,
        help="Camera identifier, e.g. C01, C02, C03",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path for output JSON (default: dataset/detections_<camera-id>.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or Path(f"dataset/detections_{args.camera_id}.json")
    det_cfg, cameras_cfg = load_config(CONFIG_PATH)
    run_detection(
        video_path=args.video,
        camera_id=args.camera_id,
        output_path=output_path,
        det_cfg=det_cfg,
        cameras_cfg=cameras_cfg,
    )


if __name__ == "__main__":
    main()
