"""
visualize.py — Phase 1.6
Overlay saved detection bounding boxes onto a video frame for visual verification.
Does NOT run YOLO — reads the JSON produced by detect.py.

Usage:
    # Save frame to dataset/visualizations/frame_<N>.jpg  (default)
    python ai_pipeline/detection/visualize.py \
        --video   dataset/raw_videos/clip_001_C01.mp4 \
        --detections dataset/detections_C01.json \
        --frame   100

    # Show in an OpenCV window instead of saving
    python ai_pipeline/detection/visualize.py \
        --video   dataset/raw_videos/clip_001_C01.mp4 \
        --detections dataset/detections_C01.json \
        --frame   100 \
        --show

    # Save to a specific path
    python ai_pipeline/detection/visualize.py \
        --video   dataset/raw_videos/clip_001_C01.mp4 \
        --detections dataset/detections_C01.json \
        --frame   100 \
        --output  dataset/visualizations/my_check.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------

BOX_COLOR    = (0, 255, 0)      # green bounding box
TEXT_COLOR   = (0, 255, 0)      # green label text
BOX_THICKNESS = 2
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.55
FONT_THICKNESS = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_detections(json_path: Path, frame_number: int) -> list[dict]:
    """Load all detection records for a specific frame from the JSON file."""
    if not json_path.exists():
        sys.exit(f"[ERROR] Detections file not found: {json_path}")
    with open(json_path) as f:
        all_detections = json.load(f)
    frame_detections = [d for d in all_detections if d["frame"] == frame_number]
    return frame_detections


def grab_frame(video_path: Path, frame_number: int):
    """Seek to frame_number in the video and return the decoded image (BGR)."""
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
        sys.exit(f"[ERROR] Could not read frame {frame_number} from {video_path}")

    return img


def draw_detections(img, detections: list[dict]) -> None:
    """Draw bounding boxes and confidence labels onto img in-place."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf = det["confidence"]
        camera_id = det.get("camera_id", "")
        timestamp = det.get("timestamp", "")

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Label: "conf: 0.94"
        label = f"conf: {conf:.2f}"
        (lw, lh), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        # Draw a filled rect behind the text so it's readable on any background
        label_y = max(y1 - 4, lh + 4)
        cv2.rectangle(
            img,
            (x1, label_y - lh - baseline - 2),
            (x1 + lw + 2, label_y + baseline - 2),
            BOX_COLOR,
            cv2.FILLED,
        )
        cv2.putText(
            img, label,
            (x1 + 1, label_y - baseline),
            FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA,
        )

    # Overlay: camera ID, timestamp, detection count (top-left corner)
    if detections:
        camera_id = detections[0].get("camera_id", "")
        timestamp  = detections[0].get("timestamp", "")
        frame_num  = detections[0].get("frame", "")
        info_lines = [
            f"Camera : {camera_id}",
            f"Frame  : {frame_num}",
            f"Time   : {timestamp}",
            f"People : {len(detections)}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(
                img, line,
                (10, 25 + i * 22),
                FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA,   # white shadow
            )
            cv2.putText(
                img, line,
                (10, 25 + i * 22),
                FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA,         # black foreground
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRACE — overlay detection boxes on a video frame (no YOLO re-run)."
    )
    parser.add_argument(
        "--video", required=True, type=Path,
        help="Path to the original video file.",
    )
    parser.add_argument(
        "--detections", required=True, type=Path,
        help="Path to the detections JSON produced by detect.py.",
    )
    parser.add_argument(
        "--frame", required=True, type=int,
        help="Zero-based frame number to visualize.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to save the output image (default: dataset/visualizations/frame_<N>.jpg).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the frame in an OpenCV window (instead of / in addition to saving).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Default save path
    output_path: Path = args.output or Path(f"dataset/visualizations/frame_{args.frame}.jpg")

    # 1. Load detections for the requested frame
    detections = load_detections(args.detections, args.frame)
    if not detections:
        print(f"[WARN] No detections found for frame {args.frame} in {args.detections}.")
        print("       The frame will still be saved/displayed without any boxes.")

    print(f"[INFO] Frame       : {args.frame}")
    print(f"[INFO] Detections  : {len(detections)} person(s)")

    # 2. Grab the raw frame from the video
    img = grab_frame(args.video, args.frame)

    # 3. Draw boxes
    draw_detections(img, detections)

    # 4. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"[INFO] Saved       : {output_path}")

    # 5. Optionally display
    if args.show:
        window_title = f"TRACE — frame {args.frame}"
        cv2.imshow(window_title, img)
        print("[INFO] Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
