"""
visualize_quality_rejections.py — Phase 3.5
Generate a visual contact sheet of quality-rejected crops for human review.

Reads:
    dataset/crop_quality_report.json  — rejection log written by crop_extractor.py
    dataset/raw_videos/clip_001_C01.mp4 — source video for extracting the frames

Writes:
    dataset/crop_verification/quality_rejections.jpg

The goal is human confirmation that:
  - crops rejected as severely_truncated really are heavily cut off,
  - no legitimate full-body crop is being discarded.

Does NOT modify any existing crop files or metadata.

Usage:
    python ai_pipeline/reid/visualize_quality_rejections.py \\
        --report  dataset/crop_quality_report.json \\
        --video   dataset/raw_videos/clip_001_C01.mp4 \\
        --output  dataset/crop_verification/quality_rejections.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Layout constants (match Phase 3.4 visualize_crops.py style)
TILE_W    = 128
TILE_H    = 256
TILE_PAD  = 8
LABEL_H   = 64
COLS      = 6
BG_COLOR  = (30, 30, 30)
TEXT_COLOR = (220, 220, 220)
RED_COLOR  = (60, 60, 200)   # BGR red — highlights rejected
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.35
FONT_THICK = 1


def fit_into_tile(img: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    """Resize preserving aspect ratio, centre on background canvas."""
    h, w   = img.shape[:2]
    scale  = min(tile_w / w, tile_h / h)
    nw     = max(1, int(w * scale))
    nh     = max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((tile_h, tile_w, 3), BG_COLOR, dtype=np.uint8)
    yo = (tile_h - nh) // 2
    xo = (tile_w - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized

    # Draw a red border to visually flag as rejected
    cv2.rectangle(canvas, (0, 0), (tile_w - 1, tile_h - 1), RED_COLOR, 2)
    return canvas


def render_label(tile_w: int, label_h: int, lines: list[str]) -> np.ndarray:
    strip = np.full((label_h, tile_w, 3), BG_COLOR, dtype=np.uint8)
    y = 13
    for line in lines[:4]:
        cv2.putText(strip, line, (2, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICK, cv2.LINE_AA)
        y += 14
    return strip


def extract_crop_from_video(
    cap: cv2.VideoCapture,
    frame_num: int,
    bbox: list,
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    """Seek to frame_num and crop the bbox region. Returns None on failure."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None

    x1 = max(0, min(int(bbox[0]), img_w))
    y1 = max(0, min(int(bbox[1]), img_h))
    x2 = max(0, min(int(bbox[2]), img_w))
    y2 = max(0, min(int(bbox[3]), img_h))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def build_contact_sheet(rejections: list[dict], cap: cv2.VideoCapture,
                        img_w: int, img_h: int) -> np.ndarray:
    """Build the rejection contact sheet from the video."""
    n    = len(rejections)
    cols = min(COLS, n)
    rows = max(1, -(-n // cols))

    cell_h = TILE_H + LABEL_H + TILE_PAD
    cell_w = TILE_W + TILE_PAD
    sheet_h = rows * cell_h + TILE_PAD
    sheet_w = cols * cell_w + TILE_PAD
    sheet   = np.full((sheet_h, sheet_w, 3), BG_COLOR, dtype=np.uint8)

    for idx, rej in enumerate(rejections):
        row = idx // cols
        col = idx % cols
        x   = TILE_PAD + col * cell_w
        y   = TILE_PAD + row * cell_h

        crop_img = extract_crop_from_video(cap, rej["frame"], rej["bbox"], img_w, img_h)

        if crop_img is None:
            tile = np.full((TILE_H, TILE_W, 3), (40, 0, 0), dtype=np.uint8)
            cv2.putText(tile, "UNREADABLE", (2, TILE_H // 2),
                        FONT, 0.35, (0, 0, 200), 1, cv2.LINE_AA)
        else:
            tile = fit_into_tile(crop_img, TILE_W, TILE_H)

        sheet[y:y + TILE_H, x:x + TILE_W] = tile

        # Label: track/frame/reason/details (truncated)
        details_short = rej["details"][:30] + "…" if len(rej["details"]) > 30 else rej["details"]
        lines = [
            f"Trk {rej['track_id']} Fr {rej['frame']}",
            rej["reason"],
            details_short,
            rej.get("timestamp", ""),
        ]
        label = render_label(TILE_W, LABEL_H, lines)
        sheet[y + TILE_H: y + TILE_H + LABEL_H, x:x + TILE_W] = label

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRACE Phase 3.5 — visualise quality-rejected crops."
    )
    parser.add_argument("--report", type=Path,
                        default=Path("dataset/crop_quality_report.json"))
    parser.add_argument("--video",  type=Path,
                        default=Path("dataset/raw_videos/clip_001_C01.mp4"))
    parser.add_argument("--output", type=Path,
                        default=Path("dataset/crop_verification/quality_rejections.jpg"))
    args = parser.parse_args()

    # Resolve relative paths against repo root
    report_path = args.report if args.report.is_absolute() else Path.cwd() / args.report
    video_path  = args.video  if args.video.is_absolute()  else Path.cwd() / args.video
    output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output

    if not report_path.exists():
        sys.exit(f"[ERROR] Quality report not found: {report_path}")
    if not video_path.exists():
        sys.exit(f"[ERROR] Video not found: {video_path}")

    with open(report_path) as f:
        report = json.load(f)

    rejections = report.get("rejections", [])
    if not rejections:
        print("[INFO] No rejections in report — nothing to visualise.")
        return

    print(f"[INFO] Visualising {len(rejections)} rejected crops from {report_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video frame size: {img_w}×{img_h}")

    sheet = build_contact_sheet(rejections, cap, img_w, img_h)
    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), sheet)
    if not ok:
        sys.exit(f"[ERROR] Failed to write output: {output_path}")

    print(f"[INFO] Contact sheet saved: {output_path}")
    print()
    print("Verify:")
    print("  - All tiles show a person strongly cut off at the frame edge.")
    print("  - No tile shows a full-body person incorrectly rejected.")
    print("  - Red borders indicate rejected status.")


if __name__ == "__main__":
    main()
