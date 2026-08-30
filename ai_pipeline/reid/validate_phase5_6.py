"""
validate_phase5_6.py — Phase 5.6
Validation Contact Sheet Generator for TRACE Re-ID

Generates easy-to-review contact sheets for manual validation of Phase 5.5
target search results. Each contact sheet shows the query image alongside
the top-K returned candidates with metadata for visual inspection.

IMPORTANT:
- This is for MANUAL validation, not automatic identity classification
- Track IDs are NOT ground-truth identity labels
- Do NOT assume same track_id = same person
- Do NOT assume different track_id = different person
- Human reviewer must visually inspect actual returned crops

Usage:
    python ai_pipeline/reid/validate_phase5_6.py \\
        --results-dir dataset/target_search \\
        --output-dir dataset/target_search/validation_sheets \\
        --query-dir dataset/query_photos

Output:
    Creates contact sheets for each query result file:
    - query_name_validation_sheet.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Track ID extraction from crop path
# ---------------------------------------------------------------------------

def extract_track_id(crop_path: str) -> Optional[int]:
    """
    Extract track_id from crop path.

    Expected format: dataset/crops_phase3_final/C01_track{track_id}_frame{frame}.jpg
    """
    try:
        # Extract from path like "dataset/crops_phase3_final/C01_track13_frame0149.jpg"
        filename = Path(crop_path).stem
        # filename is "C01_track13_frame0149"
        parts = filename.split("_")
        # parts = ["C01", "track13", "frame0149"]
        track_part = parts[1]  # "track13"
        track_id = int(track_part.replace("track", ""))
        return track_id
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Contact sheet generation
# ---------------------------------------------------------------------------

def create_validation_sheet(
    query_path: Path,
    results: List[Dict],
    output_path: Path,
    query_dir: Path,
) -> None:
    """
    Create a validation contact sheet showing query + top-K results.

    Layout:
    - Top row: Query image with label
    - Bottom rows: 5 result images in a grid with metadata labels

    Each result shows:
    - Crop image
    - Rank
    - Camera ID
    - Timestamp
    - Confidence score
    - Track ID (extracted from crop_path)
    """
    # Load query image
    query_img = Image.open(query_path).convert("RGB")

    # Load result images
    result_imgs = []
    for result in results:
        crop_path = result.get("crop_path")
        if not crop_path:
            continue

        # Resolve crop path relative to repo root
        crop_full_path = _REPO_ROOT / crop_path
        if not crop_full_path.exists():
            print(f"[WARNING] Crop not found: {crop_full_path}")
            continue

        try:
            img = Image.open(crop_full_path).convert("RGB")
            result_imgs.append((img, result))
        except Exception as e:
            print(f"[WARNING] Failed to load crop {crop_path}: {e}")

    if not result_imgs:
        print(f"[ERROR] No valid result images for query {query_path}")
        return

    # Layout configuration
    thumbnail_size = (150, 300)  # Width, height for each crop
    padding = 10
    header_height = 60

    # Calculate sheet dimensions
    # Top row: query image (scaled to fit)
    query_scale = min(300 / query_img.width, 400 / query_img.height)
    query_display_size = (int(query_img.width * query_scale), int(query_img.height * query_scale))

    # Results grid: 5 images in one row
    results_width = len(result_imgs) * thumbnail_size[0] + (len(result_imgs) + 1) * padding
    results_height = thumbnail_size[1] + 2 * padding

    total_width = max(query_display_size[0], results_width) + 2 * padding
    total_height = query_display_size[1] + header_height + results_height + 3 * padding

    # Create sheet
    sheet = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(sheet)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        header_font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        header_font = ImageFont.load_default()

    # Draw header
    y_offset = padding
    header_text = f"Query: {query_path.name}"
    draw.text((padding, y_offset), header_text, fill="black", font=header_font)
    y_offset += header_height

    # Draw query image
    query_resized = query_img.resize(query_display_size, Image.Resampling.LANCZOS)
    x_offset = padding
    sheet.paste(query_resized, (x_offset, y_offset))
    draw.rectangle([x_offset, y_offset, x_offset + query_display_size[0], y_offset + query_display_size[1]],
                   outline="blue", width=2)
    y_offset += query_display_size[1] + padding

    # Draw separator
    draw.line([(padding, y_offset), (total_width - padding, y_offset)], fill="gray", width=2)
    y_offset += padding

    # Draw results
    x_offset = padding
    for idx, (img, result) in enumerate(result_imgs, start=1):
        # Resize and paste
        img_resized = img.resize(thumbnail_size, Image.Resampling.LANCZOS)
        sheet.paste(img_resized, (x_offset, y_offset))

        # Draw border
        draw.rectangle([x_offset, y_offset, x_offset + thumbnail_size[0], y_offset + thumbnail_size[1]],
                       outline="green" if idx == 1 else "gray", width=2)

        # Extract metadata
        camera_id = result.get("camera_id", "N/A")
        timestamp = result.get("timestamp", "N/A")
        confidence = result.get("confidence", "N/A")
        crop_path = result.get("crop_path", "N/A")
        track_id = extract_track_id(crop_path)

        # Draw labels
        label_y = y_offset + thumbnail_size[1] + 2
        labels = [
            f"Rank {idx}",
            f"Cam: {camera_id}",
            f"Time: {timestamp}",
            f"Conf: {confidence}",
            f"Track: {track_id if track_id else 'N/A'}",
        ]

        for label in labels:
            draw.text((x_offset + 2, label_y), label, fill="black", font=font)
            label_y += 14

        x_offset += thumbnail_size[0] + padding

    # Save sheet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    print(f"[INFO] Saved validation sheet: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5.6 Validation Contact Sheet Generator"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing Phase 5.5 result JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save validation contact sheets"
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        required=True,
        help="Directory containing query images"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5.6 Validation Contact Sheet Generator")
    print("=" * 60)

    # Find all result JSON files
    result_files = list(args.results_dir.glob("*_top5.json"))
    if not result_files:
        print(f"[ERROR] No result files found in {args.results_dir}")
        print("[INFO] Looking for files matching pattern: *_top5.json")
        sys.exit(1)

    print(f"[INFO] Found {len(result_files)} result files")

    # Process each result file
    for result_file in result_files:
        print(f"\n[INFO] Processing: {result_file.name}")

        # Load results
        with open(result_file) as f:
            results = json.load(f)

        if not isinstance(results, list):
            print(f"[ERROR] Expected JSON list in {result_file}")
            continue

        # Determine query name from result file name
        # Expected format: track13_original_top5.json
        query_name = result_file.stem.replace("_top5", "")

        # Find query image
        # Try multiple possible locations
        possible_query_paths = [
            args.query_dir / f"{query_name}.jpg",
            args.query_dir / "query_track13" / "test_query.jpg" if "track13" in query_name else None,
            args.query_dir / "query_track16" / "test_query.jpg" if "track16" in query_name else None,
            args.query_dir / "query_track13" / f"test_query_{query_name.split('_')[-1]}.jpg" if "track13" in query_name else None,
            args.query_dir / "query_track16" / f"test_query_{query_name.split('_')[-1]}.jpg" if "track16" in query_name else None,
        ]

        query_path = None
        for possible_path in possible_query_paths:
            if possible_path and possible_path.exists():
                query_path = possible_path
                break

        if not query_path:
            print(f"[WARNING] Query image not found for {query_name}")
            print(f"[INFO] Tried: {[str(p) for p in possible_query_paths if p]}")
            continue

        # Generate output path
        output_path = args.output_dir / f"{query_name}_validation_sheet.jpg"

        # Create validation sheet
        create_validation_sheet(query_path, results, output_path, args.query_dir)

    print("\n" + "=" * 60)
    print("[INFO] Validation contact sheets generated")
    print(f"[INFO] Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
