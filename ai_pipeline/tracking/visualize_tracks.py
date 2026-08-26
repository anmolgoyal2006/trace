"""
visualize_tracks.py — Phase 2.4
Overlay ByteTrack bounding boxes and track IDs onto a video frame.
Does NOT run YOLO or ByteTrack — reads the JSON produced by track.py.

Usage:
    # Save to default path: dataset/visualizations/tracks_frame_<N>.jpg
    python ai_pipeline/tracking/visualize_tracks.py ^
        --video  dataset/raw_videos/clip_001_C01.mp4 ^
        --tracks dataset/tracks_C01.json ^
        --frame  100

    # Show in a window
    python ai_pipeline/tracking/visualize_tracks.py ^
        --video  dataset/raw_videos/clip_001_C01.mp4 ^
        --tracks dataset/tracks_C01.json ^
        --frame  100 ^
        --show

    # Custom output path
    python ai_pipeline/tracking/visualize_tracks.py ^
        --video  dataset/raw_videos/clip_001_C01.mp4 ^
        --tracks dataset/tracks_C01.json ^
        --frame  100 ^
        --output dataset/visualizations/my_tracks.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

# 20 visually distinct BGR colours — cycles if more than 20 track IDs in one frame
_PALETTE = [
    (0,   255, 0),    (0,   0,   255),  (255, 0,   0),    (0,   255, 255),
    (255, 0,   255),  (255, 255, 0),    (128, 0,   255),  (0,   128, 255),
    (0,   255, 128),  (255, 128, 0),    (128, 255, 0),    (0,   0,   128),
    (128, 0,   0),    (0,   128, 0),    (64,  0,   255),  (255, 64,  0),
    (0,   200, 200),  (200, 0,   200),  (200, 200, 0),    (100, 100, 255),
]

BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.55
FONT_THICKNESS = 1


def track_colour(track_id: int) -> tuple:
    """Return a consistent BGR colour for a given track_id."""
    return _PALETTE[track_id % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Core helpers  (same pattern as detection/visualize.py)
# ---------------------------------------------------------------------------

def load_tracks(json_path: Path, frame_number: int) -> list[dict]:
    """Load all track records for a specific frame from the JSON file."""
    if not json_path.exists():
        sys.exit(f"[ERROR] Tracks file not found: {json_path}")
    try:
        with open(json_path) as f:
            all_tracks = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] Invalid JSON in {json_path}: {e}")
    return [t for t in all_tracks if t["frame"] == frame_number]


def grab_frame(video_path: Path, frame_number: int):
    """Seek to frame_number in the video and return the decoded BGR image."""
    if not video_path.exists():
        sys.exit(f"[ERROR] Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        cap.release()
        sys.exit(
            f"[ERROR] Frame {frame_number} is out of range. "
            f"Video has {total_frames} frames (0 – {total_frames - 1})."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, img = cap.read()
    cap.release()

    if not ret or img is None:
        sys.exit(f"[ERROR] Could not decode frame {frame_number} from {video_path}")

    return img


def draw_tracks(img, tracks: list[dict]) -> None:
    """Draw bounding boxes, track ID labels, and frame info overlay in-place."""

    for trk in tracks:
        x1, y1, x2, y2 = [int(v) for v in trk["bbox"]]
        tid   = trk["track_id"]
        conf  = trk["confidence"]
        color = track_colour(tid)

        # --- bounding box ---
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # --- label: "ID: 20  conf: 0.91" ---
        label = f"ID: {tid}  conf: {conf:.2f}"
        (lw, lh), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        label_y = max(y1 - 4, lh + 4)

        # Filled background rect behind text
        cv2.rectangle(
            img,
            (x1, label_y - lh - baseline - 2),
            (x1 + lw + 4, label_y + baseline - 2),
            color,
            cv2.FILLED,
        )
        # Black text on coloured background
        cv2.putText(
            img, label,
            (x1 + 2, label_y - baseline),
            FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA,
        )

    # --- top-left info overlay ---
    if tracks:
        camera_id = tracks[0].get("camera_id", "")
        timestamp = tracks[0].get("timestamp", "")
        frame_num = tracks[0].get("frame", "")
        info_lines = [
            f"Camera : {camera_id}",
            f"Frame  : {frame_num}",
            f"Time   : {timestamp}",
            f"Tracks : {len(tracks)}",
        ]
        for i, line in enumerate(info_lines):
            # White shadow for readability on any background
            cv2.putText(img, line, (10, 25 + i * 22),
                        FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, line, (10, 25 + i * 22),
                        FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRACE — overlay ByteTrack boxes on a video frame (no model re-run)."
    )
    parser.add_argument(
        "--video", required=True, type=Path,
        help="Path to the original video file.",
    )
    parser.add_argument(
        "--tracks", required=True, type=Path,
        help="Path to tracks JSON produced by track.py.",
    )
    parser.add_argument(
        "--frame", required=True, type=int,
        help="Zero-based frame number to visualize.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output image path (default: dataset/visualizations/tracks_frame_<N>.jpg).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display in an OpenCV window (in addition to saving).",
    )
    return parser.parse_args()


def main() -> None:
    args        = parse_args()
    output_path = args.output or Path(f"dataset/visualizations/tracks_frame_{args.frame}.jpg")

    # 1. Load track records for the requested frame
    tracks = load_tracks(args.tracks, args.frame)
    if not tracks:
        print(f"[WARN] No tracks found for frame {args.frame} in {args.tracks}.")
        print("       Frame will be saved without boxes.")

    print(f"[INFO] Frame   : {args.frame}")
    print(f"[INFO] Tracks  : {len(tracks)}")
    if tracks:
        ids = [t['track_id'] for t in tracks]
        print(f"[INFO] IDs     : {ids}")

    # 2. Grab the raw frame
    img = grab_frame(args.video, args.frame)

    # 3. Draw track boxes and labels
    draw_tracks(img, tracks)

    # 4. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"[INFO] Saved   : {output_path}")

    # 5. Optionally display
    if args.show:
        cv2.imshow(f"TRACE tracks — frame {args.frame}", img)
        print("[INFO] Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
