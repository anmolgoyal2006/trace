"""
visualize_crops.py — Phase 3.4
Visual verification tool for Re-ID crop images.

Reads crops_metadata.json and generates contact sheets for human inspection.
Does NOT modify original crops or metadata.
Does NOT run YOLO, ByteTrack, or Re-ID.

Modes:
  --num-random N     random N crops contact sheet → random_samples.jpg
  --track-id   ID    full track contact sheet     → track_<ID>.jpg
  --random-track     pick one track randomly      → track_<ID>.jpg

Usage:
  python ai_pipeline/reid/visualize_crops.py \
      --metadata dataset/crops_metadata.json \
      --num-random 12 \
      --output-dir dataset/crop_verification

  python ai_pipeline/reid/visualize_crops.py \
      --metadata dataset/crops_metadata.json \
      --track-id 37 \
      --output-dir dataset/crop_verification

  python ai_pipeline/reid/visualize_crops.py \
      --metadata dataset/crops_metadata.json \
      --random-track \
      --output-dir dataset/crop_verification \
      --show
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Contact sheet layout constants
TILE_W       = 128   # max tile width  (px) — aspect ratio preserved
TILE_H       = 256   # max tile height (px)
TILE_PAD     = 8     # padding between tiles (px)
COLS         = 6     # tiles per row
LABEL_H      = 52    # pixels reserved below each tile for text
BG_COLOR     = (30, 30, 30)    # dark background
TEXT_COLOR   = (220, 220, 220) # light text
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.38
FONT_THICK   = 1


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_metadata(metadata_path: Path) -> list[dict]:
    if not metadata_path.exists():
        sys.exit(f"[ERROR] Metadata file not found: {metadata_path}")
    with open(metadata_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit("[ERROR] Metadata must be a JSON list.")
    return data


def abs_crop_path(record: dict) -> Path:
    """Resolve a metadata crop_path to an absolute Path."""
    cp = record["crop_path"]
    p  = Path(cp)
    return p if p.is_absolute() else (REPO_ROOT / p)


# ---------------------------------------------------------------------------
# Contact sheet builder
# ---------------------------------------------------------------------------

def fit_into_tile(img: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    """
    Resize img to fit within (tile_w × tile_h) preserving aspect ratio.
    Returns a new tile_w × tile_h image with the crop centred on BG_COLOR.
    """
    h, w = img.shape[:2]
    scale = min(tile_w / w, tile_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((tile_h, tile_w, 3), BG_COLOR, dtype=np.uint8)
    y_off  = (tile_h - new_h) // 2
    x_off  = (tile_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def render_label(tile_w: int, label_h: int, lines: list[str]) -> np.ndarray:
    """Render up to 3 text lines into a (label_h × tile_w) strip."""
    strip = np.full((label_h, tile_w, 3), BG_COLOR, dtype=np.uint8)
    y = 13
    for line in lines[:3]:
        cv2.putText(strip, line, (2, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICK, cv2.LINE_AA)
        y += 16
    return strip


def make_contact_sheet(
    records:   list[dict],
    cols:      int  = COLS,
    tile_w:    int  = TILE_W,
    tile_h:    int  = TILE_H,
    label_h:   int  = LABEL_H,
    pad:       int  = TILE_PAD,
) -> tuple[np.ndarray, dict]:
    """
    Build a contact sheet from the given metadata records.

    Returns:
        sheet   — BGR numpy image
        stats   — dict with unreadable/dark/tiny/wide counts
    """
    stats = {"unreadable": 0, "dark": 0, "tiny": 0, "wide": 0, "total": 0}

    cell_h = tile_h + label_h + pad
    cell_w = tile_w + pad
    rows   = max(1, -(-len(records) // cols))   # ceiling division

    sheet_h = rows * cell_h + pad
    sheet_w = cols * cell_w + pad
    sheet   = np.full((sheet_h, sheet_w, 3), BG_COLOR, dtype=np.uint8)

    for idx, rec in enumerate(records):
        stats["total"] += 1
        row = idx // cols
        col = idx  % cols
        x   = pad + col * cell_w
        y   = pad + row * cell_h

        crop_path = abs_crop_path(rec)
        img = cv2.imread(str(crop_path)) if crop_path.exists() else None

        if img is None:
            stats["unreadable"] += 1
            placeholder = np.full((tile_h, tile_w, 3), (60, 0, 0), dtype=np.uint8)
            cv2.putText(placeholder, "MISSING", (4, tile_h // 2),
                        FONT, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
            tile = placeholder
        else:
            h, w = img.shape[:2]
            if img.mean() < 10:
                stats["dark"] += 1
            if h < 60 or w < 30:
                stats["tiny"] += 1
            if w > h * 2:
                stats["wide"] += 1
            tile = fit_into_tile(img, tile_w, tile_h)

        # Paste tile
        sheet[y:y + tile_h, x:x + tile_w] = tile

        # Label lines
        line1 = f"{rec.get('camera_id','')} | Trk {rec.get('track_id','')} | Fr {rec.get('frame','')}"
        line2 = rec.get("timestamp", "")
        line3 = f"conf={rec.get('detection_confidence', rec.get('confidence', '?')):.2f}"
        label_strip = render_label(tile_w, label_h, [line1, line2, line3])
        ly = y + tile_h
        sheet[ly:ly + label_h, x:x + tile_w] = label_strip

    return sheet, stats


# ---------------------------------------------------------------------------
# Sanity summary
# ---------------------------------------------------------------------------

def print_sanity_summary(records: list[dict], stats: dict) -> None:
    print()
    print("=== Automated Sanity Checks ===")
    print(f"  Total crops inspected : {stats['total']}")
    print(f"  Unreadable crops      : {stats['unreadable']}")
    print(f"  Very dark crops       : {stats['dark']}")
    print(f"  Unusually tiny crops  : {stats['tiny']}")
    print(f"  Unusually wide crops  : {stats['wide']}")
    print()
    print("  NOTE: These checks do NOT prove crops are correctly cropped.")
    print("        Human inspection of the contact sheet is required.")


# ---------------------------------------------------------------------------
# Mode: random samples
# ---------------------------------------------------------------------------

def mode_random(records: list[dict], n: int, seed: int,
                output_dir: Path, show: bool) -> None:
    rng = random.Random(seed)
    chosen = rng.sample(records, min(n, len(records)))

    print(f"[INFO] Random mode — {len(chosen)} crops (seed={seed})")
    sheet, stats = make_contact_sheet(chosen)

    out = output_dir / "random_samples.jpg"
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"[INFO] Saved: {out}")

    print_sanity_summary(chosen, stats)

    if show:
        cv2.imshow("TRACE — random crop samples", sheet)
        print("[INFO] Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Mode: full track
# ---------------------------------------------------------------------------

def mode_track(records: list[dict], track_id: int,
               output_dir: Path, show: bool) -> None:
    track_recs = sorted(
        [r for r in records if r.get("track_id") == track_id],
        key=lambda r: r["frame"],
    )

    if not track_recs:
        sys.exit(f"[ERROR] No records found for track_id={track_id}.")

    frames = [r["frame"] for r in track_recs]
    camera = track_recs[0].get("camera_id", "?")

    print("=" * 44)
    print("Track Visual Verification")
    print("=" * 44)
    print(f"Camera         : {camera}")
    print(f"Track ID       : {track_id}")
    print(f"Sampled crops  : {len(track_recs)}")
    print(f"Frames         : {frames}")
    print()

    sheet, stats = make_contact_sheet(track_recs, cols=min(5, len(track_recs)))

    out = output_dir / f"track_{track_id}.jpg"
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"[INFO] Saved: {out}")
    print()
    print(f"Open {out} and verify:")
    print("  - All crops contain a visible person.")
    print("  - Clothing/appearance is consistent across frames.")
    print("  - No crop shows an obviously different person.")
    print("  - No crop is blank, dark, or badly cut off.")

    print_sanity_summary(track_recs, stats)

    if show:
        cv2.imshow(f"TRACE — track {track_id}", sheet)
        print("[INFO] Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Mode: random track
# ---------------------------------------------------------------------------

def mode_random_track(records: list[dict], seed: int,
                      output_dir: Path, show: bool) -> None:
    track_ids = sorted({r["track_id"] for r in records})
    rng       = random.Random(seed)
    chosen_id = rng.choice(track_ids)
    print(f"[INFO] Random-track mode — selected track_id={chosen_id} (seed={seed})")
    mode_track(records, chosen_id, output_dir, show)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRACE — visual verification of Re-ID crop images."
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=Path("dataset/crops_metadata.json"),
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("dataset/crop_verification"))
    parser.add_argument("--num-random", type=int, default=None,
                        help="Random-sample mode: number of crops to show.")
    parser.add_argument("--track-id", type=int, default=None,
                        help="Full-track mode: track ID to inspect.")
    parser.add_argument("--random-track", action="store_true",
                        help="Pick one track randomly.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42).")
    parser.add_argument("--show", action="store_true",
                        help="Display result in an OpenCV window.")
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    meta_p  = args.metadata if args.metadata.is_absolute() \
              else (Path.cwd() / args.metadata)
    records = load_metadata(meta_p)

    if args.num_random is not None:
        mode_random(records, args.num_random, args.seed, args.output_dir, args.show)
    elif args.track_id is not None:
        mode_track(records, args.track_id, args.output_dir, args.show)
    elif args.random_track:
        mode_random_track(records, args.seed, args.output_dir, args.show)
    else:
        print("[INFO] No mode specified. Use --num-random, --track-id, or --random-track.")
        print("       Run with --help for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
