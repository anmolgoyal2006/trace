"""
Phase 5.4.1 — Visual Track Inspection / Contact Sheet Generator
================================================================
Given one or more track IDs, finds every crop belonging to each track
in the Phase 3 metadata, and saves a labelled contact sheet for visual
inspection.

Designed for diagnostic comparison of Track 66 (highest Phase 5.3
competitor) vs Track 29 (known source track) to understand whether the
similarity gap is caused by visual confusion, tracker ID collision, or
another factor.

Usage examples
--------------
Single track:
    python ai_pipeline/reid/inspect_track.py --track-id 66

Two tracks:
    python ai_pipeline/reid/inspect_track.py --track-id 66 29

All three outputs (also dumps a per-track similarity line if provided):
    python ai_pipeline/reid/inspect_track.py --track-id 66 29 \
        --metadata dataset/crops_metadata_phase3_final.json \
        --output-dir dataset/target_search/inspection

Output
------
  dataset/target_search/inspection/track_66_contact_sheet.jpg
  dataset/target_search/inspection/track_29_contact_sheet.jpg

Each sheet shows every crop as a tile at a uniform display height,
with a label strip below each tile showing:
  track_id  |  frame  |  timestamp  |  filename  |  conf  |  orig size
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT    = Path(__file__).resolve().parents[2]
_META_DEFAULT = _REPO_ROOT / "dataset" / "crops_metadata_phase3_final.json"
_OUT_DEFAULT  = _REPO_ROOT / "dataset" / "target_search" / "inspection"

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

TILE_HEIGHT      = 300    # px — all crops are scaled to this height, width proportional
TILE_MAX_WIDTH   = 200    # px — cap very wide tiles
LABEL_HEIGHT     = 68     # px — strip below each tile for text labels
LABEL_FONT_SIZE  = 11     # pt
TILE_PAD         = 8      # px — horizontal gap between tiles
MARGIN           = 16     # px — border around the whole sheet
HEADER_HEIGHT    = 36     # px — title bar at the top
BG_COLOUR        = (30, 30, 30)       # dark background
TILE_BG          = (50, 50, 50)       # slightly lighter tile background
HEADER_BG        = (20, 20, 80)       # deep blue header
LABEL_BG         = (40, 40, 40)
TEXT_COLOUR      = (230, 230, 230)
HEADER_TEXT      = (255, 255, 180)
BORDER_COLOUR    = (80, 80, 80)
MISSING_COLOUR   = (180, 40, 40)      # red placeholder for unreadable crops
JPEG_QUALITY     = 92


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CropRecord(NamedTuple):
    crop_path:            str          # relative path (as in metadata)
    full_path:            Path         # absolute
    track_id:             int
    frame:                int
    timestamp:            str
    detection_confidence: float
    bbox:                 list


class LoadResult(NamedTuple):
    record:  CropRecord
    image:   Image.Image | None        # None if unreadable
    error:   str | None                # error message when image is None


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_metadata(meta_path: Path) -> list[dict]:
    if not meta_path.exists():
        sys.exit(f"[ERROR] Metadata file not found: {meta_path}")
    with open(meta_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        sys.exit(f"[ERROR] Expected a JSON list in {meta_path}")
    return data


def find_crops_for_track(
    metadata: list[dict],
    track_id: int,
    repo_root: Path,
) -> list[CropRecord]:
    """Return CropRecords for every metadata entry with the given track_id,
    sorted ascending by frame number."""
    records = []
    for entry in metadata:
        if entry.get("track_id") != track_id:
            continue
        rel = entry["crop_path"]
        full = repo_root / rel.replace("/", os.sep)
        records.append(CropRecord(
            crop_path=rel,
            full_path=full,
            track_id=int(entry["track_id"]),
            frame=int(entry["frame"]),
            timestamp=str(entry.get("timestamp", "")),
            detection_confidence=float(entry.get("detection_confidence", 0.0)),
            bbox=entry.get("bbox", []),
        ))
    records.sort(key=lambda r: r.frame)
    return records


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_crop(record: CropRecord) -> LoadResult:
    """Attempt to open a crop image.  Returns LoadResult with image=None on failure."""
    if not record.full_path.exists():
        return LoadResult(record=record, image=None,
                          error=f"File not found: {record.full_path}")
    try:
        img = Image.open(record.full_path)
        img.verify()                      # detect truncation / corruption
    except (UnidentifiedImageError, Exception) as exc:
        return LoadResult(record=record, image=None,
                          error=f"Cannot verify: {exc}")
    try:
        img = Image.open(record.full_path).convert("RGB")
        return LoadResult(record=record, image=img, error=None)
    except Exception as exc:
        return LoadResult(record=record, image=None,
                          error=f"Cannot open/convert: {exc}")


# ---------------------------------------------------------------------------
# Font helper — falls back to default if TrueType not available
# ---------------------------------------------------------------------------

def _get_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Tile rendering
# ---------------------------------------------------------------------------

def _scale_to_height(img: Image.Image, target_h: int, max_w: int) -> Image.Image:
    """Scale img so its height == target_h, capping width at max_w."""
    orig_w, orig_h = img.size
    scale  = target_h / orig_h
    new_w  = min(int(orig_w * scale), max_w)
    new_h  = target_h
    return img.resize((new_w, new_h), Image.LANCZOS)


def _make_error_tile(
    record: CropRecord,
    error: str,
    tile_w: int = 120,
) -> Image.Image:
    """Red placeholder tile for unreadable crops."""
    h = TILE_HEIGHT + LABEL_HEIGHT
    tile = Image.new("RGB", (tile_w, h), TILE_BG)
    draw = ImageDraw.Draw(tile)
    font = _get_font(LABEL_FONT_SIZE)
    # red block
    draw.rectangle([0, 0, tile_w - 1, TILE_HEIGHT - 1], fill=MISSING_COLOUR)
    draw.text((4, TILE_HEIGHT // 2 - 10), "UNREADABLE", fill=(255, 255, 255), font=font)
    # label area
    draw.rectangle([0, TILE_HEIGHT, tile_w - 1, h - 1], fill=LABEL_BG)
    label = f"frame={record.frame}\n{os.path.basename(record.crop_path)}\n{error[:30]}"
    draw.text((4, TILE_HEIGHT + 2), label, fill=MISSING_COLOUR, font=font)
    return tile


def _make_crop_tile(load_result: LoadResult) -> Image.Image:
    """Render one tile: scaled image + label strip."""
    rec   = load_result.record
    img   = load_result.image

    if img is None:
        return _make_error_tile(rec, load_result.error or "unknown error")

    scaled  = _scale_to_height(img, TILE_HEIGHT, TILE_MAX_WIDTH)
    tile_w  = scaled.width
    tile_h  = TILE_HEIGHT + LABEL_HEIGHT

    tile = Image.new("RGB", (tile_w, tile_h), TILE_BG)
    tile.paste(scaled, (0, 0))

    # label strip
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, TILE_HEIGHT, tile_w - 1, tile_h - 1], fill=LABEL_BG)
    draw.line([(0, TILE_HEIGHT), (tile_w, TILE_HEIGHT)], fill=BORDER_COLOUR, width=1)

    font  = _get_font(LABEL_FONT_SIZE)
    orig_w, orig_h = img.size
    lines = [
        f"track={rec.track_id}  frame={rec.frame}",
        f"ts: {rec.timestamp}",
        f"conf: {rec.detection_confidence:.3f}  orig: {orig_w}x{orig_h}",
        os.path.basename(rec.crop_path),
    ]
    y = TILE_HEIGHT + 3
    for line in lines:
        draw.text((3, y), line, fill=TEXT_COLOUR, font=font)
        y += LABEL_FONT_SIZE + 3

    return tile


# ---------------------------------------------------------------------------
# Contact sheet assembly
# ---------------------------------------------------------------------------

def make_contact_sheet(
    track_id:     int,
    load_results: list[LoadResult],
    output_path:  Path,
    query_marker: str | None = None,
) -> Path:
    """
    Assemble a contact sheet from a list of LoadResults and save it.

    Parameters
    ----------
    track_id:      The track ID being shown.
    load_results:  One LoadResult per crop, in frame order.
    output_path:   Where to save the JPEG.
    query_marker:  Optional string appended to the title (e.g. 'SOURCE TRACK').

    Returns
    -------
    Path to the saved file.
    """
    if not load_results:
        print(f"  [WARN] No crops found for track {track_id} — skipping contact sheet")
        return output_path

    # Render all tiles
    tiles = [_make_crop_tile(r) for r in load_results]

    tile_widths  = [t.width  for t in tiles]
    tile_heights = [t.height for t in tiles]   # all same height: TILE_HEIGHT + LABEL_HEIGHT

    total_tile_w = sum(tile_widths) + TILE_PAD * (len(tiles) - 1)
    tile_h       = max(tile_heights)

    sheet_w = total_tile_w + 2 * MARGIN
    sheet_h = tile_h + 2 * MARGIN + HEADER_HEIGHT

    sheet = Image.new("RGB", (sheet_w, sheet_h), BG_COLOUR)
    draw  = ImageDraw.Draw(sheet)

    # Header bar
    draw.rectangle([0, 0, sheet_w, HEADER_HEIGHT], fill=HEADER_BG)
    header_font = _get_font(14)
    n_ok   = sum(1 for r in load_results if r.image is not None)
    n_fail = len(load_results) - n_ok
    title  = f"Track {track_id}  —  {len(load_results)} crops  ({n_ok} OK, {n_fail} unreadable)"
    if query_marker:
        title += f"  [{query_marker}]"
    draw.text((MARGIN, 8), title, fill=HEADER_TEXT, font=header_font)

    # Paste tiles
    x = MARGIN
    y = HEADER_HEIGHT + MARGIN
    for tile in tiles:
        sheet.paste(tile, (x, y))
        # thin border around each tile
        draw.rectangle(
            [x - 1, y - 1, x + tile.width, y + tile.height],
            outline=BORDER_COLOUR,
        )
        x += tile.width + TILE_PAD

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(output_path), format="JPEG", quality=JPEG_QUALITY)
    return output_path


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_track_summary(
    track_id:     int,
    records:      list[CropRecord],
    load_results: list[LoadResult],
) -> None:
    n_ok   = sum(1 for r in load_results if r.image is not None)
    n_fail = len(load_results) - n_ok
    print(f"\nTrack {track_id}: {len(records)} crops found, {n_ok} readable, {n_fail} unreadable")
    for lr in load_results:
        rec = lr.record
        orig_size = (
            f"{lr.image.size[0]}x{lr.image.size[1]}"
            if lr.image else "N/A"
        )
        status = "OK   " if lr.image else "FAIL "
        err    = f"  ← {lr.error}" if lr.error else ""
        print(
            f"  [{status}] frame={rec.frame:04d}  ts={rec.timestamp}"
            f"  conf={rec.detection_confidence:.3f}  size={orig_size}"
            f"  {os.path.basename(rec.crop_path)}{err}"
        )


# ---------------------------------------------------------------------------
# Public API (also used by tests)
# ---------------------------------------------------------------------------

def inspect_tracks(
    track_ids:   list[int],
    meta_path:   Path | None = None,
    output_dir:  Path | None = None,
    repo_root:   Path | None = None,
    source_track: int | None = None,
) -> dict[int, Path]:
    """
    Load crops for each requested track_id and save contact sheets.

    Parameters
    ----------
    track_ids:    List of track IDs to process.
    meta_path:    Path to crops_metadata_phase3_final.json.
                  Defaults to the repository default.
    output_dir:   Directory where contact sheets are saved.
                  Defaults to dataset/target_search/inspection/.
    repo_root:    Repository root for resolving crop paths.
                  Defaults to the two-levels-up parent of this file.
    source_track: If set, the contact sheet title for this track is
                  labelled 'SOURCE TRACK'.

    Returns
    -------
    Dict mapping track_id -> Path of the saved contact sheet.
    """
    if repo_root is None:
        repo_root = _REPO_ROOT
    if meta_path is None:
        meta_path = _META_DEFAULT
    if output_dir is None:
        output_dir = _OUT_DEFAULT

    metadata = load_metadata(meta_path)
    saved_paths: dict[int, Path] = {}

    for tid in track_ids:
        records = find_crops_for_track(metadata, tid, repo_root)
        if not records:
            print(f"[WARN] Track {tid}: no crops found in metadata — skipping")
            continue

        load_results = [load_crop(r) for r in records]
        _print_track_summary(tid, records, load_results)

        marker = "SOURCE TRACK" if tid == source_track else None
        out_path = output_dir / f"track_{tid}_contact_sheet.jpg"
        saved = make_contact_sheet(tid, load_results, out_path, query_marker=marker)
        print(f"  → Contact sheet saved: {saved}")
        saved_paths[tid] = saved

    return saved_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TRACE Phase 5.4.1 — Visual Track Inspection\n"
            "Generates a labelled contact sheet for each requested track."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--track-id",
        nargs="+",
        type=int,
        required=True,
        metavar="ID",
        help="One or more track IDs to inspect (e.g. --track-id 66 29)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=_META_DEFAULT,
        help=f"Path to crops_metadata_phase3_final.json (default: {_META_DEFAULT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_OUT_DEFAULT,
        help=f"Output directory for contact sheets (default: {_OUT_DEFAULT})",
    )
    parser.add_argument(
        "--source-track",
        type=int,
        default=29,
        help="Track ID to label as 'SOURCE TRACK' on its contact sheet (default: 29)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("=" * 60)
    print("TRACE Phase 5.4.1 — Visual Track Inspection")
    print("=" * 60)
    print(f"  Track IDs    : {args.track_id}")
    print(f"  Metadata     : {args.metadata}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Source track : {args.source_track}")
    print()

    saved = inspect_tracks(
        track_ids=args.track_id,
        meta_path=args.metadata,
        output_dir=args.output_dir,
        source_track=args.source_track,
    )

    print("\n" + "=" * 60)
    print(f"Contact sheets generated: {len(saved)}")
    for tid, path in sorted(saved.items()):
        print(f"  Track {tid:>4}  →  {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
